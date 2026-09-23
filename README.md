# 整数采样数字锁相环(PLL)捕获诊断

基于合成复数导频的二阶离散 PLL 捕获模块,提供逐样本诊断、失锁冻结、
任意分块一致处理与严格的输入校验。

## 环境

- Python 3.14.7,Linux 原生运行
- 仅使用标准库(`math` / `dataclasses` / `unittest` / `json`),**无需安装任何第三方依赖**
- 核心、测试、演示全部离线;示例数据由仓库内确定性代码生成

## 算法契约

每个有效样本按如下顺序更新(状态为相位 `theta`、频率字 `omega`):

1. 旋转: `y = z * exp(-j*theta)`
2. 误差: `error = wrap(atan2(Im y, Re y))`,取值 `[-pi, pi)`(`atan2` 恰为 `pi` 时映射为 `-pi`)
3. 频率: `omega' = clip(omega + ki*error, -wmax, +wmax)`(触发限幅在诊断中标记 `saturated`)
4. 相位: `theta' = wrap(theta + omega' + kp*error)`

单位约定:

- `theta` / `error` / `eps`:物理相位,单位 **rad**;
- `omega` / `wmax`:频率字,单位 **rad/sample**,即物理角频率 `Omega`(rad/s) 乘以采样周期 `Ts`;
- `kp` / `ki`:每样本增益(无量纲);
- 物理频率换算:`f = omega * fs / (2*pi)` Hz,`fs = 1/Ts`。

捕获与失锁:

- 幅度 `|z| < amin` 的样本为**无效样本**:冻结 `omega`/`theta` 更新,清零捕获计数,
  累计一次失锁(`lost_samples`),诊断中 `valid=False`、`error=None`。
- 连续 `lock_count`(L)个**有效**样本满足 `|error| <= eps` 才进入 `locked`;
  任一有效样本 `|error| > eps` 立即清零计数并退出 `locked`。
- 参数稳定范围在构造时强制检查:`0 < kp < 1` 且 `0 < ki < kp`。

## 运行中切换采样周期

`pll.set_sample_period(new_period)` 在运行中切换采样周期 `Ts`(秒),按
`r = Ts_new / Ts_old` 同步缩放频率字与增益,保持**物理相位连续**与连续时间环路动态不变:

| 状态/参数 | 缩放 | 保持不变的物理量 |
|---|---|---|
| `omega` | `omega * r` | 物理角频率 `Omega = omega / Ts` |
| `kp` | `kp * r` | 连续时间比例增益 `kp / Ts` |
| `ki` | `ki * r^2` | 连续时间积分增益 `ki / Ts^2` |
| `wmax` | `wmax * r` | 限幅对应的物理频率范围 |
| `theta`、`eps`、`amin`、`lock_count`、捕获计数、失锁计数 | 不变 | 物理相位(rad)连续 |

- 构造时可用 `PLL(params, sample_period=...)` 指定初始周期(默认 `1.0`,即归一化
  rad/sample,旧调用行为不变);`pll.sample_period` / `pll.sample_rate` 只读查询。
- 切换成功返回 `SamplePeriodChange` 报告:`old_period, new_period, ratio,
  omega_before, omega_after, theta, old_params, new_params`,及不变量属性
  `angular_frequency`(rad/s)。
- 切换是**原子**的:先用缩放后的值完成全部参数校验,`new_period` 非正/非数值,或缩放后
  越界(如 `kp*r >= 1`、`ki*r^2 >= kp*r`)时抛 `PLLParamError`,PLL 状态保持调用前原样。
- 切换后 `run()` 结果快照携带 `sample_period`;`result.frequency_offset()` 省略参数时
  按跟踪的周期换算 Hz(显式传 `sample_rate` 的旧用法不变)。

## 接口

```python
from pll_capture import PLL, PLLParams, PLLResult, Diagnostic, SamplePeriodChange

params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03, lock_count=40)
pll = PLL(params, theta0=0.0, omega0=0.0)   # |omega0| > wmax 会被拒绝而非静默限幅
diag = pll.step(z)                           # 单样本 -> Diagnostic
result = pll.run(samples)                    # 批量 -> PLLResult(含本批逐样本诊断)
result.omega                                 # 估计频偏 rad/sample
result.frequency_offset(48000.0)             # 换算 Hz: omega * fs / (2*pi)

pll48 = PLL(params, sample_period=1/48000)   # 跟踪采样周期
pll48.run(samples48k)
change = pll48.set_sample_period(1/24000)    # 运行中切换: 48 kHz -> 24 kHz
change.angular_frequency                     # 切换前后不变的物理角频率 rad/s
result2 = pll48.run(samples24k)
result2.frequency_offset()                   # 省略参数时按跟踪的周期换算 Hz
```

- `PLLParams(kp, ki, wmax, amin, eps, lock_count)`:构造时校验,`int` 数值统一规范为 `float`。
- `PLL(params, theta0=0.0, omega0=0.0, sample_period=1.0)`:`theta0` 任意有限实数(内部 wrap 到 `[-pi, pi)`);`sample_period` 为有限正数(秒)。
- `Diagnostic`:`index, valid, amplitude, error, omega, theta, streak, locked, saturated`。
- `PLLResult`:`params, omega, theta, locked, streak, lost_samples, processed, diagnostics, sample_period`。
- 同一实例可任意分块连续调用 `run()`/`step()`,与一次性处理**逐位一致**(浮点运算顺序相同)。

## 错误语义

所有错误均为 `ValueError` 子类,消息中带参数名/样本值,可定位:

- `PLLParamError`:环路参数或初始状态非法(如 `kp >= 1`、`ki >= kp`、`lock_count` 非 int、
  `|omega0| > wmax`、`sample_period` 非正),以及采样周期切换后缩放参数越界。
- `PLLInputError`:样本非法(bool、非数值、NaN/Infinity)、批次不可迭代、单批超上限。

校验规则:数值输入拒绝 NaN/Infinity;整数参数(`lock_count`)拒绝 bool 与非 int;
样本拒绝 bool。非法输入一律抛错,**不静默纠正**。

**批次原子性**:`run()` 先整批校验再逐样本更新;任一样本非法时抛 `PLLInputError`,
PLL 状态保持调用前原样,不留部分变更。`set_sample_period()` 同样先校验后提交:
非法周期或缩放后参数越界时抛 `PLLParamError`,状态(参数、omega、theta、周期)原样保留。

## 输入上限

- `run()` 单批最多 `MAX_SAMPLES = 1_000_000` 个样本(诊断逐样本保存,限制内存;可分块处理更长的流)。
- 诊断内存占用为 O(本批样本数)。

## 误差口径(浮点)

- 所有计算为 IEEE 754 双精度;同一实例分块与整批处理结果**逐位相等**(测试以 `==` 断言)。
- 参考测试(`tests/data/*.json`)的期望值是导频的理论真值(生成相位/频偏),不是被测核心的输出:
  - 恒相位参考:收敛后 `|omega| <= 1e-9`、`|wrap(theta - 0.6)| <= 1e-9`、末端 `|error| <= 1e-9`;
  - 固定频偏参考:收敛后 `|omega - 0.3| <= 1e-9`、末端 `|error| <= 1e-6`。
- 采样周期切换测试的期望值来自解析关系(`omega*r`、`kp*r`、`ki*r^2`、`wmax*r`,
  物理角频率 `omega/Ts` 不变)与信号模型 `z = exp(j*(phi0 + Omega*t))` 的直接采样,
  不经过被测实现;切换后相位误差稳态残差 `< 1e-4`(远小于 `eps`),`omega` 收敛到
  `Omega*Ts_new` 的容差为 `1e-9`。
- 收敛为指数渐近,上述容差对应数百个样本后的稳态残差量级,不用于掩盖边界行为
  (限幅、冻结、计数清零均以精确条件断言)。

## 设计取舍

- **先校验后更新**:`run()` 牺牲一次遍历换取批次原子性,避免有状态对象的部分变更。
- **切换缩放律 `kp*r, ki*r^2`**:把每样本增益还原为连续时间增益(`Kp = kp/Ts`,
  `Ki = ki/Ts^2`)并保持其不变,使切换前后环路对同一物理信号的动态响应一致;
  `omega` 与 `wmax` 按 `r` 线性缩放保持物理角频率不变,`theta` 为物理相位天然连续。
- **切换不清零捕获计数**:缩放保持物理相位连续,锁定状态在物理上未被打断,故
  `streak`/`locked`/`lost_samples` 原样保留。
- **无效样本清零捕获计数**:契约要求"连续 L 个有效样本",无效样本打断连续性,故同时退出 locked。
- **`omega0` 超限拒绝而非限幅**:初始状态属于调用方显式输入,静默修改会掩盖配置错误。
- **旋转用 `cos/sin` 而非 `cmath.exp`**:等价且避免复数构造开销,仅依赖 `math`。

## 已知限制

- 仅实现题面契约的二阶环路(无环路滤波器变体、无噪声建模、无辅助捕获/扫频)。
- 切换采样周期只重缩放环路状态与增益;调用方需自行保证切换后的样本流确实来自
  新周期(本模块不感知样本的实际到达时刻)。上采样(`r` 大)受稳定性约束
  `kp*r < 1` 且 `ki*r < kp` 限制,越界即拒绝。
- `wmax` 不限制在 `pi` 以内;但采样系统频偏超过 `pi` rad/sample 会混叠,环路无法捕获,这属于信号条件而非缺陷。
- 不宣称兼容任何行业标准的完整 PLL 规范。
- 诊断全量保存在内存中,超长流需调用方自行分块并丢弃历史诊断。

## 运行

```bash
python -m unittest discover -s tests -v   # 单元测试(28 项)
python demo.py                            # 捕获过程 + 采样周期切换 + 非法输入拒绝, 约 0.1 秒
```
