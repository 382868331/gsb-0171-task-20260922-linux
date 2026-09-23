# 整数采样数字锁相环(PLL)捕获诊断

基于合成复数导频的二阶离散 PLL 捕获模块,提供逐样本诊断、失锁冻结、
任意分块一致处理、严格输入校验,以及**运行中切换采样周期**(自动缩放增益
与频率状态以保持物理相位连续)。

## 环境

- Python 3.14.7,Linux 原生运行
- 仅使用标准库(`math` / `dataclasses` / `unittest` / `json`),**无需安装任何第三方依赖**
- 核心、测试、演示全部离线;示例数据由仓库内确定性代码生成

## 单位约定

| 量 | 单位 | 说明 |
|---|---|---|
| 相位 `theta`、相位误差 `error`、门限 `eps` | rad | `error` 取 `[-pi, pi)`(`atan2` 恰为 `pi` 时映射为 `-pi`) |
| 频率字 `omega`、限幅 `wmax` | rad/sample | 每个采样间隔的相位增量 |
| 增益 `kp` | rad/rad(无量纲) | 比例相位步进系数 |
| 增益 `ki` | rad/(rad·sample)=1/sample | 频率积分系数 |
| 采样周期 `ts` | 秒(s) | `SamplePeriod.seconds(Ts)`;不关心物理时间时用默认单位周期 |
| 采样率 `fs` | Hz | `fs = 1/Ts` |
| 物理频偏 | Hz | `f = omega / (2*pi*Ts) = omega*fs/(2*pi)` |

采样点上的信号相位模型:`phi_k = phi0 + omega*k`(omega 为该速率下的
rad/sample);物理时间 `t_k = k*Ts` 时 `phi_k = phi0 + 2*pi*f*t_k`。

## 算法契约

每个有效样本按如下顺序更新(状态为相位 `theta`(rad)、频率字 `omega`
(rad/sample)):

1. 旋转: `y = z * exp(-j*theta)`
2. 误差: `error = wrap(atan2(Im y, Re y))`,取值 `[-pi, pi)`
3. 频率: `omega' = clip(omega + ki*error, -wmax, +wmax)`(触发限幅在诊断中标记 `saturated`)
4. 相位: `theta' = wrap(theta + omega' + kp*error)`

捕获与失锁:

- 幅度 `|z| < amin` 的样本为**无效样本**:冻结 `omega`/`theta` 更新,清零捕获计数,
  累计一次失锁(`lost_samples`),诊断中 `valid=False`、`error=None`。
- 连续 `lock_count`(L)个**有效**样本满足 `|error| <= eps` 才进入 `locked`;
  任一有效样本 `|error| > eps` 立即清零计数并退出 `locked`。
- 参数稳定范围在构造时强制检查:`0 < kp < 1` 且 `0 < ki < kp`。

## 接口

```python
from pll_capture import (
    PLL, PLLParams, PLLResult, Diagnostic,
    SamplePeriod, RateSwitchReport,
)

# 旧固定速率接口:不传 ts 即单位采样周期,行为与历史版本完全一致。
params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03, lock_count=40)
pll = PLL(params, theta0=0.0, omega0=0.0)   # |omega0| > wmax 会被拒绝而非静默限幅
diag = pll.step(z)                           # 单样本 -> Diagnostic
result = pll.run(samples)                    # 批量 -> PLLResult(含本批逐样本诊断)
result.omega                                 # 估计频偏 rad/sample
result.frequency_offset(48000.0)             # 旧换算入口: omega * fs / (2*pi), 单位 Hz

# 带物理采样周期(秒)的构造:
Ts = 1.0 / 48000.0
params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03,
                   lock_count=40, ts=SamplePeriod.seconds(Ts))
pll = PLL(params)
pll.sample_rate          # 48000.0 Hz(只读属性)
pll.frequency_hz         # omega / (2*pi*Ts), 当前物理频偏 Hz
```

- `SamplePeriod.seconds(Ts)`:采样周期(秒),有限正数;`SamplePeriod()` 为默认
  单位周期(1.0)。不另设 Hz 构造器:`SamplePeriod.seconds(1/fs)`。
- `PLLParams(..., ts=...)`:`ts` 仅用于物理量换算与切换缩放,**逐样本更新方程
  与数值结果不因是否传 `ts` 而改变**(旧构造方式与旧结果逐位一致)。
- `Diagnostic`:`index, valid, amplitude, error, omega, theta, streak, locked, saturated`。
- `PLLResult`:`params, omega, theta, locked, streak, lost_samples, processed, diagnostics`。
- 同一实例可任意分块连续调用 `run()`/`step()`,与一次性处理**逐位一致**(浮点运算顺序相同)。

### 运行中切换采样周期 `PLL.set_sample_period()`

```python
report = pll.set_sample_period(SamplePeriod.seconds(1.0 / 24000.0))
# 便捷形式(等价): pll.set_sample_period(1.0 / 24000.0)
```

切换在**上一个样本之后、下一个样本之前**生效(切换本身不消耗样本;下一个样本
距上一个样本为一个新周期)。设 `r = Ts_new / Ts_old`,一次性原子完成:

- `theta_new = theta_old`(rad 数值不变,切换瞬间**物理相位连续**);
- `omega_new = r * omega_old`(rad/sample 随周期缩放,**物理频偏 Hz 不变**);
- `kp_new = r*kp`、`ki_new = r^2*ki`、`wmax_new = r*wmax`。

增益缩放的依据是线性化环路的物理时间域不变量(由更新方程的前向欧拉离散化
反推):自然频率 `omega_n = sqrt(ki)/Ts`(rad/s)、阻尼比
`zeta = kp/(2*sqrt(ki))`。代入缩放后二者在切换前后严格相等,故物理收敛速度
(按秒计)与阻尼不变。`eps`、`amin`、`lock_count` 是相位量/样本计数,不随
采样周期改变;`streak`、`locked`、`lost_samples`、`processed` 也保持不变。
锁定状态下切换时,由于 `theta` 连续且 `omega_new` 恰好匹配信号在新速率下的
每样本相位增量,下一个样本误差仍约为 0,锁定不中断。

返回 `RateSwitchReport`(全部为切换前后快照,可核对):
`index, period_old, period_new, ratio, params_old, params_new, theta_old,
theta_new, omega_old, omega_new, physical_frequency_hz`。

边界与拒绝:

- `r = 1`(相同周期)为合法幂等操作,状态逐位不变;
- 若 `kp*r >= 1` 或 `ki*r^2 >= kp*r`(新速率下增益破坏稳定范围),抛
  `PLLSwitchError`,**不做任何状态变更**——物理上这对应新速率过低、
  前向欧拉离散化已不稳定,需要调用方显式选择参数;
- `omega` 已处于限幅时:`|omega_new| = r*wmax = wmax_new`,恰好落在新限幅上,合法。

## 错误语义

所有错误均为 `ValueError` 子类(共同基类 `PLLError`),消息中带参数名/样本值,可定位:

- `PLLParamError`:环路参数或初始状态非法(如 `kp >= 1`、`ki >= kp`、`lock_count` 非 int、
  `|omega0| > wmax`、`ts` 不是 `SamplePeriod`)。
- `PLLInputError`:样本非法(bool、非数值、NaN/Infinity)、批次不可迭代、单批超上限。
- `PLLSwitchError`:运行中切换采样周期非法(周期非有限正数、单位非法、
  或缩放后增益不满足稳定范围)。

校验规则:数值输入拒绝 NaN/Infinity;整数参数(`lock_count`)拒绝 bool 与非 int;
样本拒绝 bool。非法输入一律抛错,**不静默纠正**。

**批次原子性**:`run()` 先整批校验再逐样本更新;任一样本非法时抛 `PLLInputError`,
PLL 状态保持调用前原样,不留部分变更。`set_sample_period()` 同理:先全部校验
(含缩放后稳定性)再一次性提交,失败时参数、频率、相位、计数全部保持原值。

## 输入上限

- `run()` 单批最多 `MAX_SAMPLES = 1_000_000` 个样本(诊断逐样本保存,限制内存;可分块处理更长的流)。
- 诊断内存占用为 O(本批样本数)。

## 误差口径(浮点)

- 所有计算为 IEEE 754 双精度;同一实例分块与整批处理结果**逐位相等**(测试以 `==` 断言)。
- 切换缩放 `*r`/`*r^2` 为普通双精度乘法;`r` 为 2 的整数次幂(如 48kHz→24kHz)
  时缩放是精确的,其他比值存在末位舍入(约 1e-16 相对量级)。
- 参考测试(`tests/data/*.json`)的期望值是导频的理论真值(生成相位/频偏),不是被测核心的输出:
  - 恒相位参考:收敛后 `|omega| <= 1e-9`、`|wrap(theta - 0.6)| <= 1e-9`、末端 `|error| <= 1e-9`;
  - 固定频偏参考:收敛后 `|omega - 0.3| <= 1e-9`、末端 `|error| <= 1e-6`。
- 切换测试的期望值由物理关系独立推导:Hz 不变、`omega_new/omega_old = r`、
  `theta_new == theta_old`、`omega_n`/`zeta` 解析不变量、切换后首个新速率样本
  的误差按信号真值独立计算,不调用被测核心生成期望。
- 收敛为指数渐近,上述容差对应数百个样本后的稳态残差量级,不用于掩盖边界行为
  (限幅、冻结、计数清零、切换拒绝均以精确条件断言)。

## 设计取舍

- **先校验后更新**:`run()` 牺牲一次遍历换取批次原子性;`set_sample_period()`
  先算出并校验全部新值再提交,避免有状态对象的部分变更。
- **切换时缩放增益而非保持数字增益不变**:题面要求保持物理相位连续与物理动态;
  保持 `omega_n`(rad/s)、`zeta` 不变是离散二阶环速率换算的标准关系。
- **`theta` 不重采样、不跳转**:切换约定在样本之间生效,切换瞬间就是相位对齐点,
  保持数值不变即物理连续。
- **缩放后不稳定直接拒绝**:大比率降采样会使前向欧拉离散化失去稳定性
  (`kp*r >= 1`),拒绝比静默产生发散环路更可定位。
- **无效样本清零捕获计数**:契约要求"连续 L 个有效样本",无效样本打断连续性,故同时退出 locked。
- **`omega0` 超限拒绝而非限幅**:初始状态属于调用方显式输入,静默修改会掩盖配置错误。
- **旋转用 `cos/sin` 而非 `cmath.exp`**:等价且避免复数构造开销,仅依赖 `math`。

## 已知限制

- 仅实现题面契约的二阶环路(无环路滤波器变体、无噪声建模、无辅助捕获/扫频/导频搜索)。
- 切换速率时不做缺样复锁、不重放或重采样历史样本;假设调用方按新周期提供后续样本。
- 无完整状态保存/恢复(序列化)接口;`RateSwitchReport` 只保存单次切换的快照。
- 增益缩放保持的是连续化物理动态;离散化本身有 O(Ts) 量级速率相关偏差,
  极大比率切换后收敛轨迹不会与"全程单一速率"逐点相同(稳态真值不变)。
- `wmax` 不限制在 `pi` 以内;但采样系统频偏超过 `pi` rad/sample 会混叠,环路无法捕获,这属于信号条件而非缺陷。
- 不宣称兼容任何行业标准的完整 PLL 规范。
- 诊断全量保存在内存中,超长流需调用方自行分块并丢弃历史诊断。

## 运行

```bash
python -m unittest discover -s tests -v   # 单元测试
python demo.py                            # 捕获 + 采样周期切换演示 + 边界/非法拒绝, 数秒内完成
```
