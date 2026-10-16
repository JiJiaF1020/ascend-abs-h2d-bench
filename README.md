# Ascend 950 H2D 带宽测试集

两个工具：

1. `test_abs_h2d_bandwidth.py` —— `torch.abs` 全连续 host DRAM→HBM 带宽，外加
   `copy_` / pinned DMA 基线。**纯 torch_npu，不需要编译**。
2. `test_scatter_contig_vs_random.py` —— C8 scatter 搬运 kernel 的**离散 vs 连续
   对照实验**（同一 kernel、同一 656B 行粒度、同一深度 8 MTE 队列，只改地址模式）。
   **需要先编译自定义算子**（`bash build_c8.sh`）。

## 工具 1：abs / copy / pinned H2D 带宽（免编译）

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
- copy 基线：`out.copy_(host)` 同口径数据（`op=copy`），用于对比 swapped DRAM 的
  拷贝路径，可用 `--no-copy-baseline` 关掉。
- pinned 基线：`pin_memory=True` 锁页内存 → HBM 的 `copy_(non_blocking=True)`
  （`op=pinned_copy`），走 DMA 拷贝引擎，是**全连续 H2D 的硬件上限探针**，
  可用 `--no-pinned-baseline` 关掉。三种 op 对比可定位瓶颈：abs（计算 kernel 直读）
  vs copy（swapped 拷贝路径）vs pinned_copy（DMA 引擎）。

## 单卡与多卡

默认只测 `--devices npu:0` **一个 die**。多 die 有两种测法：

```bash
# 顺序扫：每个 die 单独测（互不争抢，看单 die 最佳能力）
python3 test_abs_h2d_bandwidth.py --devices npu:0 npu:1 npu:2 npu:3 npu:4 npu:5 npu:6 npu:7 \
  --json-output results/abs_h2d_seq8.json

# 并发测 8 卡：每 die 一个进程同时跑（共享 host DRAM/PCIe，与旧 8 卡绑核图同口径）
python3 run_multidie.py --devices 0 1 2 3 4 5 6 7 \
  --json-output results/abs_h2d_8die.json
# 只测大 size 省时：参数放 -- 后面转发给单卡脚本
python3 run_multidie.py -- --dtypes fp16 --sizes-mb 512 1024 2048 --iters 50
```

并发版输出每个 (op, dtype, size) 的 8 die 带宽**之和**（`sum_bw_gbps`），可直接和旧图
240-260 GB/s 对比；每 die 的明细在 `results/multidie_parts/die*.json`，日志在同名 `.log`。

注意与旧 8 卡图对比口径：旧图 240-260 GB/s 是 8 die 带宽之和，单 die 约 30 GB/s，
和本测试单 die 的数值是同一量级。

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

## 工具 2：scatter 搬运 kernel 离散 vs 连续对照（需编译）

回答的问题：**同一个搬运 kernel，离散访问和连续访问的 H2D 带宽差多少。**
同一 `kvcache_scatter_copy_c8` kernel（656B 行、深度 8 MTE2 队列、QUEUE_DEPTH=8），
只把元数据从随机（randperm block table / src / dst）换成连续（恒等 block table、
顺序 src/dst），其余变量完全一致。

```bash
# 编译（一次即可；需要 CANN 环境的 msopgen）
export ASCEND_HOME_PATH=/usr/local/Ascend/cann   # 按实际路径替换
source "$ASCEND_HOME_PATH/set_env.sh"
export A5_SOC_VERSION=ascend950 NANOVLLM_A5_OPS_BUILD_JOBS=64
bash build_c8.sh
# 通过标准：末尾打印 "[c8_build] torch extension import: OK"

# 运行 A/B 对照（单 die；默认 copy_count 扫 300/2048/8192/16384，bs=24）
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
python3 test_scatter_contig_vs_random.py \
  --device npu:0 --json-output results/scatter_ab_npu0.json
```

输出每个 (pattern, copy_count) 一行 `A5_SCATTER_AB_RESULT`（含逐次迭代统计），
结尾打印对照表：`copy_count / random_gbps / contig_gbps / contig/random` 比值。
每个配置计时前后都做全量逐字节校验（poison 填充 + byte-exact 比较 +
copy_count=0 不动 HBM 检查）。

带宽口径与工具 1 相同：`copied_tokens × 656 / avg_us / 1000`（GB/s，只计一次有效
payload）。注意：连续模式下每核按 `coreIdx + k×usedCoreNum` 交错取行，单核流是
跨核交错的，整 die 视角才是全连续覆盖——这正是对照实验想要的（两模式只有地址
模式不同）。

### 八卡跑 scatter A/B 对照

```bash
# 先完成 build_c8.sh（只需一次），然后 8 die 并发：
python3 run_multidie.py --worker scatter --devices 0 1 2 3 4 5 6 7 \
  --json-output results/scatter_ab_8die.json
# 省时版：只测大 copy 档
python3 run_multidie.py --worker scatter -- --copy-counts 2048 8192 16384 --iters 100
```

聚合表按 (pattern, copy_count) 输出 8 die 带宽之和，random 与 contiguous 两行直接
可比；每 die 明细在 `results/multidie_parts/die*.json`。
