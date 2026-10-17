# Ascend 950 H2D 带宽基准测试

host DRAM → HBM 搬运带宽的系统性实测，回答三个问题：

1. **算子（kernel）路径的搬运极限是多少** —— 算子内由 AI Core 直接读主机内存；
2. **离散 vs 连续访问差多少** —— 同一搬运 kernel，只改地址模式；
3. **DMA 引擎 vs kernel 路径差多少** —— pinned memory `copy_` vs 任何 kernel。

测试机：Ascend950DT(9582)，单卡 1 die，整机 16 NPU（CPU0 带 NPU 0-7，CPU1 带 NPU 8-15）。

## 结论与数据

**kernel 路径存在约 22 GB/s/die 的架构性上限，与访问模式、chunk 大小、算子实现无关；
DMA 引擎路径达 50.4 GB/s/die，是 kernel 路径的 2.3 倍。8 die 并发时每 die 仍卡在
自身的 22 GB/s 帽上；1:4 拓扑（两 socket 分担）比 1:8（单 socket 带 8 卡）高 18%。**

单 die(host DRAM → HBM，大 payload 稳态）:

| 路径 | 带宽 (GB/s) | 说明 |
|---|---|---|
| `torch.abs`(swapped 输入 → HBM 输出) | 22.1 | 通用计算 kernel 直读 host |
| `VllmA5H2dContigCopy`(本仓自定义算子） | 21~22 | 专用搬运 kernel：16KB DataCopyPad × 深度 4 MTE 队列，全连续 |
| scatter kernel 离散 vs 连续 | 比值 0.99~1.02 | 同一 kernel 只改地址模式，**无差异** |
| **`pinned_copy`(pinned memory → HBM `copy_`)** | **50.4** | DMA 拷贝引擎，Gen5 x16 链路真实能力 |

8 die 并发（`VllmA5H2dContigCopy`，合计带宽 GB/s):

| 拓扑 | 1024 MiB | 2048 MiB | 单 die 折算 |
|---|---|---|---|
| 1:8(NPU 0-7 全在 CPU0) | 151.9 | 151.9 | ~19.0 |
| 1:4(两 socket 各 4) | 180.1 | 179.7 | ~22.5 |

解读：单 die 内访问模式无所谓（深度队列隐藏随机性）；整机层面瓶颈是每 die 的
kernel 路径帽，其次才是单 socket 内存/PCIe 上行的共享争抢（1:8 vs 1:4 差 18%)。
规划容量时：算子内搬运按 ~22 GB/s/die，主机侧 DMA 预取按 ~50 GB/s/die。

## 仓库内容

| 文件 | 说明 |
|---|---|
| `test_abs_h2d_bandwidth.py` | 工具 1：`torch.abs` + `copy_` + pinned DMA 三条路径，免编译 |
| `test_scatter_contig_vs_random.py` | 工具 2：scatter kernel 离散/连续对照（需编译） |
| `test_h2d_contig_copy.py` | 工具 3：`VllmA5H2dContigCopy` 纯连续 DataCopyPad 搬运 kernel（需编译） |
| `run_multidie.py` | 多 die 并发启动器：`--worker abs\|scatter\|contig`，逐配置计时屏障 + NUMA 绑核 |
| `build_c8.sh` / `csrc/` / `torch_extension/` | 自定义算子（scatter + contig）构建脚手架 |

通用口径：带宽 = 有效 payload ÷ 平均耗时，只计一次 H2D 方向；逐次迭代 NPU Event
计时（保留 min/max/p50/p95/原始序列），wall clock 交叉校验；计时前后做字节级校验。

## 快速开始

```bash
# 环境（ASCEND_HOME_PATH 指向含 set_env.sh 的目录）
export ASCEND_HOME_PATH=/usr/local/Ascend/cann   # 按实际路径替换
source "$ASCEND_HOME_PATH/set_env.sh"
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"

# 工具 1：免编译，直接跑（含 DMA pinned_copy 基线）
python3 test_abs_h2d_bandwidth.py --device npu:0 --sizes-mb 1024 2048 \
  --dtypes fp16 --no-copy-baseline --json-output results/abs_npu0.json

# 工具 2/3：先编译一次自定义算子
export A5_SOC_VERSION=ascend950 NANOVLLM_A5_OPS_BUILD_JOBS=64
bash build_c8.sh   # 通过标准：末尾 "[c8_build] torch extension import: OK"
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH

python3 test_h2d_contig_copy.py --device npu:0 --json-output results/contig_1die.json
python3 test_scatter_contig_vs_random.py --device npu:0 --json-output results/scatter_ab_1die.json
```

## 多 die 并发

`run_multidie.py` 每 die 一个进程同时跑，逐配置计时屏障对齐（所有 die 到齐才计时），
可选 NUMA 绑核；聚合表输出各 die 带宽之和，明细在 `results/multidie_parts/`。

```bash
# 1:8：8 个 NPU 全在 CPU0
python3 run_multidie.py --worker contig --devices 0 1 2 3 4 5 6 7 \
  --numa-nodes 0 0 0 0 0 0 0 0 --json-output results/contig_8die_1to8.json

# 1:4：两个 socket 各 4 个 NPU
python3 run_multidie.py --worker contig --devices 0 2 4 6 8 10 12 14 \
  --numa-nodes 0 0 0 0 1 1 1 1 --json-output results/contig_8die_1to4.json
```

NUMA 节点编号以 `lscpu | grep -i numa` 实际输出为准；机器没有 `numactl` 时去掉
`--numa-nodes` 即可（不绑核）。`--` 后的参数原样转发给单卡脚本，例如
`-- --sizes-mb 512 1024 2048 --iters 100`。

## 故障

- 编译报错：确认 CANN 9.1+ 且已 `source set_env.sh`(`msopgen` 在 PATH 里）。
- 运行报 `torch_npu.empty_with_swapped_memory is unavailable`：torch_npu 版本太旧。
- 算子调用报错：确认 `PYTHONPATH` 指向本仓 `torch_extension` 且 `build_c8.sh` 成功。
