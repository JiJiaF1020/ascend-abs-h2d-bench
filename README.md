# Ascend 950 abs 全连续 H2D 最大带宽测试

用 `torch.abs(host_tensor, out=device_tensor)` 实测单 die 上**全连续访问**的
DRAM→HBM（H2D 路径）最大有效带宽。

- **输入**：1D 全连续大张量，`torch_npu.empty_with_swapped_memory` 分配（真实主机 DRAM）。
- **输出**：同形 HBM 张量（普通 `torch.empty(device=...)`）。
- **纯 torch_npu 内置算子，不需要编译任何东西**，单文件脚本，无第三方依赖。

## 运行方法

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann   # 若本机 CANN 路径不同请替换
source "$ASCEND_HOME_PATH/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=0

python3 test_abs_h2d_bandwidth.py \
  --device npu:0 --sizes-mb 64 256 1024 2048 --warmup 5 --iters 50 \
  --json-output results/abs_fp16_sizes64-2048_cards1.json
```

可选对照：

```bash
# fp32（同字节数下元素数减半）
python3 test_abs_h2d_bandwidth.py --dtype fp32 --sizes-mb 1024 2048

# 更大 payload 探饱和上限（注意 host DRAM 占用约 2×size）
python3 test_abs_h2d_bandwidth.py --sizes-mb 4096 8192 --warmup 3 --iters 20
```

## 输出与口径

终端打印每个 size 的 `A5_ABS_H2D_RESULT` 行、三列结果表（size_mb / avg_us /
bandwidth_gbps）和 `A5_ABS_H2D_MAX_BANDWIDTH` 最大值行；`--json-output` 保存同口径 JSON。

带宽口径：`payload_bytes / avg_us / 1000`（GB/s），**只计一次有效 payload**
（host→device 方向读取字节数），HBM 侧输出写入不重复计数。计时为一对 NPU Event
包住 `iters` 次连续 launch 后取平均；swapped-memory 初始化（`fill_`/`add_` staging）
不在计时区间。每档计时前会抽样校验 abs 结果与 CPU 参考逐字节一致。

通过标志：日志出现 `A5_ABS_H2D_CONFIG ...`、`A5_ABS_H2D_MAX_BANDWIDTH ... ok=1`、
`A5_ABS_H2D_UT_OK`。

## 故障

若报 `torch.abs(swapped_memory_input, out=hbm_output) failed`，说明该 torch_npu
版本不接受 swapped-memory 输入配 HBM 输出——请原样保留报错，需要换自定义 kernel
方案（kernel 内直接读 host GM 地址）。
