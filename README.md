# Ascend 950 abs 全连续 H2D 最大带宽测试

用 `torch.abs(host_tensor, out=device_tensor)` 实测 **全连续访问**的
DRAM→HBM（H2D 路径）有效带宽，并扫出最大值。

- **输入**：1D 全连续大张量，`torch_npu.empty_with_swapped_memory` 分配（真实主机 DRAM）。
- **输出**：同形 HBM 张量（普通 `torch.empty(device=...)`）。
- **纯 torch_npu 内置算子，不需要编译任何东西**，单文件脚本，无第三方依赖。

## 运行方法

先找 CANN 路径（三种办法，找到一个即可）：

```bash
ls /usr/local/Ascend/                                    # 默认安装位置
env | grep -i ascend                                     # 之前配过环境的话直接在
find /usr/local /home /opt -name "set_env.sh" -path "*Ascend*" 2>/dev/null
```

`ASCEND_HOME_PATH` 指向包含 `set_env.sh` 的目录，典型两种：
`/usr/local/Ascend/cann`（合并安装/容器）或 `/usr/local/Ascend/ascend-toolkit/latest`
（标准 toolkit 安装）。

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann   # 按上面找到的实际路径替换
source "$ASCEND_HOME_PATH/set_env.sh"
# 验证环境：能打印版本号即 OK
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 默认全量：sizes 1..2048 MiB 对数 12 档 × fp16/bf16/fp32 × abs + copy 基线
python3 test_abs_h2d_bandwidth.py \
  --devices npu:0 \
  --json-output results/abs_h2d_full_npu0.json
```

常用调节：

```bash
# 多 die 顺序扫（每 die 独立全套数据）
python3 test_abs_h2d_bandwidth.py --devices npu:0 npu:1 npu:2 npu:3 --json-output results/abs_h2d_cards4.json

# 只看大 payload 饱和段，加大迭代数
python3 test_abs_h2d_bandwidth.py --sizes-mb 512 1024 2048 4096 8192 --iters 50

# 只要 abs，不要 copy 基线
python3 test_abs_h2d_bandwidth.py --no-copy-baseline
```

## 采集的数据

每个 (device × op × dtype × size) 配置一行结果，包含：

- `avg_us` / `bandwidth_gbps`：逐次迭代 Event 计时的均值及换算带宽。
- `wall_us` / `wall_bandwidth_gbps`：wall clock 交叉校验口径（含 launch 间隙）。
- `per_iter_us` / `bandwidth_per_iter_gbps`：min / max / mean / std / p50 / p95。
- `raw_iter_us`：**每一次迭代的原始耗时**，全部保留在 JSON 里。
- 环境信息：hostname、python/torch/torch_npu 版本、CANN 路径、`npu-smi info` 原文。
- copy 基线：`out.copy_(host)` 同口径数据（`op=copy`），用于对比纯 H2D 拷贝路径上限，
  可用 `--no-copy-baseline` 关掉。

`--json-output` 同时生成同目录 `.csv` 汇总表（每配置一行，便于画图）。

## 带宽口径

`payload_bytes / avg_us / 1000`（GB/s），**只计一次有效 payload**
（host→device 方向读取字节数），HBM 侧输出写入不重复计数。计时为逐次迭代
NPU Event；swapped-memory 初始化（`fill_`/`add_` staging）不在计时区间。
每档计时前会抽样校验结果与 CPU 参考逐字节一致（abs 校验 abs、copy 校验原值）。

通过标志：每配置一行 `A5_ABS_H2D_RESULT ... ok=1`，结尾
`A5_ABS_H2D_MAX_BANDWIDTH ... ok=1` 与 `A5_ABS_H2D_UT_OK`。

## 故障

若报 `... with swapped-memory input and HBM output failed`，说明该 torch_npu
版本不接受 swapped-memory 输入配 HBM 输出——请原样保留报错，需要换自定义 kernel
方案（kernel 内直接读 host GM 地址）。

## 备注

- 主机内存占用约为每档 2×size（CPU 引用张量 + swapped 张量），8192 MiB 档约 16 GiB，
  注意 host DRAM 余量。
- 默认全套（3 dtype × 12 size × abs+copy）单 die 大约几分钟，主要花在 CPU 侧随机数初始化。
