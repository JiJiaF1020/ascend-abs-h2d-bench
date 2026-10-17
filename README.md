# Ascend 950 H2D 带宽基准测试

host DRAM → HBM 搬运带宽测试集。三条路径可对比：

- **kernel 路径**：算子内由 AI Core 直接读主机内存（`torch.abs` 或本仓自定义搬运算子）；
- **DMA 引擎路径**：pinned memory → HBM `copy_`；
- **离散 vs 连续**：同一搬运 kernel 只改地址模式的对照实验。

输入张量经 `torch_npu.empty_with_swapped_memory` 分配（真实主机 DRAM）。

## 仓库内容

| 文件 | 说明 |
|---|---|
| `test_abs_h2d_bandwidth.py` | 工具 1：`torch.abs` + `copy_` + pinned DMA 三条路径，免编译 |
| `test_scatter_contig_vs_random.py` | 工具 2：scatter kernel 离散/连续对照（需编译） |
| `test_h2d_contig_copy.py` | 工具 3：`VllmA5H2dContigCopy` 纯连续 DataCopyPad 搬运 kernel（需编译） |
| `run_multidie.py` | 多 die 并发启动器：`--worker abs\|scatter\|contig`，逐配置计时屏障对齐 |
| `build_c8.sh` / `csrc/` / `torch_extension/` | 自定义算子（scatter + contig）构建脚手架 |

通用口径：带宽 = 有效 payload ÷ 平均耗时，只计一次 H2D 方向；逐次迭代 NPU Event
计时（保留 min/max/p50/p95/原始序列），wall clock 交叉校验；计时前后做字节级校验。

## 环境

```bash
# ASCEND_HOME_PATH 指向含 set_env.sh 的目录；找不到可以这样搜：
find /usr/local /home /opt -name "set_env.sh" -path "*Ascend*" 2>/dev/null

export ASCEND_HOME_PATH=/usr/local/Ascend/cann   # 按实际路径替换
source "$ASCEND_HOME_PATH/set_env.sh"
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
```

## 工具 1：abs / copy / pinned DMA（免编译）

```bash
python3 test_abs_h2d_bandwidth.py --device npu:0 \
  --sizes-mb 64 256 1024 2048 --dtypes fp16 \
  --json-output results/abs_npu0.json
```

参数：`--devices`(顺序多 die)、`--sizes-mb`、`--dtypes fp16|bf16|fp32`、
`--warmup`、`--iters`、`--no-copy-baseline`、`--no-pinned-baseline`。

## 工具 2/3：自定义搬运算子（需编译一次）

```bash
export A5_SOC_VERSION=ascend950 NANOVLLM_A5_OPS_BUILD_JOBS=64
bash build_c8.sh   # 通过标准：末尾 "[c8_build] torch extension import: OK"
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH

# 纯连续 DataCopyPad 搬运（默认 1 / 1024 / 2048 MiB）
python3 test_h2d_contig_copy.py --device npu:0 --json-output results/contig_1die.json

# scatter kernel 离散 vs 连续对照
python3 test_scatter_contig_vs_random.py --device npu:0 --json-output results/scatter_ab_1die.json
```

## 多 die 并发

每个 die 一个进程同时跑，逐配置计时屏障（所有 die 到齐才一起计时），聚合表输出
各 die 带宽之和；每 die 明细和日志在 `results/multidie_parts/`。

```bash
# 16 NPU 机器示例：1:8（NPU 0-7 全在 CPU0 一侧）
python3 run_multidie.py --worker contig --devices 0 1 2 3 4 5 6 7 \
  --json-output results/contig_8die_1to8.json

# 1:4（两个 socket 各 4 个 NPU）
python3 run_multidie.py --worker contig --devices 0 2 4 6 8 10 12 14 \
  --json-output results/contig_8die_1to4.json

# `--` 后的参数原样转发给单卡脚本
python3 run_multidie.py --worker contig -- --sizes-mb 512 1024 2048 --iters 100
```

## 故障

- 编译报错：确认 CANN 9.1+ 且已 `source set_env.sh`(`msopgen` 在 PATH 里）。
- 运行报 `torch_npu.empty_with_swapped_memory is unavailable`：torch_npu 版本太旧。
- 算子调用报错：确认 `PYTHONPATH` 指向本仓 `torch_extension` 且 `build_c8.sh` 成功。
- 绝对带宽数值与机型相关（CPU 的 PCIe 根复合体/内存子系统不同），跨机型比较
  绝对值前请先确认两台机器的链路代次一致。
