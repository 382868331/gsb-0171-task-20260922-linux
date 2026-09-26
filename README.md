# 整数采样数字锁相环(PLL)捕获诊断(含 holdover/reacquire)

基于合成复数导频的二阶离散 PLL 捕获模块,提供逐样本诊断、失锁冻结、
任意分块一致处理与严格的输入校验;可选的 tracking / holdover / reacquire
三态状态机用于处理导频短暂消失。

## 环境

- Python 3.14.7,Linux 原生运行
- 仅使用标准库(`math` / `dataclasses` / `unittest` / `json`),**无需安装任何第三方依赖**
- 核心、测试、演示全部离线;示例数据由仓库内确定性代码生成

## 算法契约

每个有效样本按如下顺序更新(状态为相位 `theta`、频率字 `omega`,单位均为 rad/sample):

1. 旋转: `y = z * exp(-j*theta)`
2. 误差: `error = wrap(atan2(Im y, Re y))`,取值 `[-pi, pi)`(`atan2` 恰为 `pi` 时映射为 `-pi`)
3. 频率: `omega' = clip(omega + ki*error, -wmax, +wmax)`(触发限幅在诊断中标记 `saturated`)
4. 相位: `theta' = wrap(theta + omega' + kp*error)`

捕获与失锁:

- 幅度 `|z| < amin` 的样本为**无效样本**:冻结 `omega`/`theta` 更新,清零捕获计数,
  累计一次失锁(`lost_samples`),诊断中 `valid=False`、`error=None`。
- 连续 `lock_count`(L)个**有效**样本满足 `|error| <= eps` 才进入 `locked`;
  任一有效样本 `|error| > eps` 立即清零计数并退出 `locked`。
- 参数稳定范围在构造时强制检查:`0 < kp < 1` 且 `0 < ki < kp`。

## 导频短暂消失:三态状态机(可选,默认关闭)

默认参数(`holdover_samples=0`)下行为与旧版完全一致。设置
`holdover_samples >= 1` 后启用三态状态机:

- **tracking**:正常跟踪,更新方程同上。
- **holdover**:幅度连续低于 `amin` 达到 `holdover_samples` 个样本后进入。
  冻结最近 `median_window` 个有效样本频率估计的**中位数** `omega_hold`,
  每样本仅以 `theta += omega_hold` 外推相位,**不再用噪声更新积分项**
  (omega 保持冻结值)。进入 holdover 之前的低幅样本仍按旧版语义冻结。
- **reacquire**:参考重新有效(幅度 >= amin)后进入。**不瞬间重置相位**:
  每样本相位校正限幅在 `±max_phase_step` 内
  (`theta' = wrap(theta + omega' + clip(kp*error, ±max_phase_step))`),
  逐渐追上参考。连续 `reacquire_count` 个有效样本**同时**满足
  相位误差条件 `|error| <= eps` 与频率误差条件 `|omega' - omega| <= freq_eps`
  才重新标记 `locked` 并回到 tracking;任一条件不满足即清零重捕获计数。
  reacquire 期间再次出现低幅样本会按 `holdover_samples` 计数重新进入 holdover。

**门限相等时的归属(明确约定)**:

- 幅度 `== amin`:判为**有效**样本(无效判定是严格的 `<`);
- `|error| == eps`:计入捕获/重捕获计数(`<=` 语义);
- `|omega' - omega| == freq_eps`:计入重捕获计数(`<=` 语义)。

低幅计数与重捕获计数都是 PLL 实例的内部状态,在不同 `run()`/`step()`
调用(不同 chunk)之间延续。

**诚实口径**:holdover 期间参考已丢失,PLL 无法知道真实相位,只能用冻结的
中位频率外推。诊断与 demo 报告的是外推相位与真实相位的**保持误差**,
不宣称参考丢失后仍能跟踪真值。

## 接口

```python
from pll_capture import PLL, PLLParams, PLLResult, Diagnostic

params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03, lock_count=40)
# 启用三态状态机(全部新参数均有默认值,旧代码无需修改):
params2 = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03, lock_count=40,
                    holdover_samples=8, median_window=11, max_phase_step=0.05,
                    freq_eps=0.01, reacquire_count=25)
pll = PLL(params, theta0=0.0, omega0=0.0)   # |omega0| > wmax 会被拒绝而非静默限幅
diag = pll.step(z)                           # 单样本 -> Diagnostic
result = pll.run(samples)                    # 批量 -> PLLResult(含本批逐样本诊断)
result.omega                                 # 估计频偏 rad/sample
result.frequency_offset(48000.0)             # 换算 Hz: omega * fs / (2*pi)
pll.state                                    # "tracking" / "holdover" / "reacquire"
```

- `PLLParams(kp, ki, wmax, amin, eps, lock_count, holdover_samples=0,`
  `median_window=5, max_phase_step=0.05, freq_eps=0.01, reacquire_count=10)`:
  构造时校验,`int` 数值统一规范为 `float`;新参数均可选,默认值保持旧行为。
- `PLL(params, theta0=0.0, omega0=0.0)`:`theta0` 任意有限实数(内部 wrap 到 `[-pi, pi)`)。
- `Diagnostic`:`index, valid, amplitude, error, omega, theta, streak, locked,`
  `saturated, state`(`state` 为末尾新增字段,带默认值)。
- `PLLResult`:`params, omega, theta, locked, streak, lost_samples, processed,`
  `diagnostics, state`。
- 同一实例可任意分块连续调用 `run()`/`step()`,与一次性处理**逐位一致**(浮点运算顺序相同)。

## 错误语义

所有错误均为 `ValueError` 子类,消息中带参数名/样本值,可定位:

- `PLLParamError`:环路参数或初始状态非法(如 `kp >= 1`、`ki >= kp`、`lock_count` 非 int、
  `|omega0| > wmax`、`holdover_samples` 为负、`max_phase_step <= 0` 等)。
- `PLLInputError`:样本非法(bool、非数值、NaN/Infinity)、批次不可迭代、单批超上限。

校验规则:数值输入拒绝 NaN/Infinity;整数参数(`lock_count`)拒绝 bool 与非 int;
样本拒绝 bool。非法输入一律抛错,**不静默纠正**。

**批次原子性**:`run()` 先整批校验再逐样本更新;任一样本非法时抛 `PLLInputError`,
PLL 状态保持调用前原样,不留部分变更。

## 输入上限

- `run()` 单批最多 `MAX_SAMPLES = 1_000_000` 个样本(诊断逐样本保存,限制内存;可分块处理更长的流)。
- 诊断内存占用为 O(本批样本数)。

## 误差口径(浮点)

- 所有计算为 IEEE 754 双精度;同一实例分块与整批处理结果**逐位相等**(测试以 `==` 断言)。
- 参考测试(`tests/data/*.json`)的期望值是导频的理论真值(生成相位/频偏),不是被测核心的输出:
  - 恒相位参考:收敛后 `|omega| <= 1e-9`、`|wrap(theta - 0.6)| <= 1e-9`、末端 `|error| <= 1e-9`;
  - 固定频偏参考:收敛后 `|omega - 0.3| <= 1e-9`、末端 `|error| <= 1e-6`。
- 收敛为指数渐近,上述容差对应数百个样本后的稳态残差量级,不用于掩盖边界行为
  (限幅、冻结、计数清零均以精确条件断言)。

## 设计取舍

- **先校验后更新**:`run()` 牺牲一次遍历换取批次原子性,避免有状态对象的部分变更。
- **无效样本清零捕获计数**:契约要求"连续 L 个有效样本",无效样本打断连续性,故同时退出 locked。
- **`omega0` 超限拒绝而非限幅**:初始状态属于调用方显式输入,静默修改会掩盖配置错误。
- **旋转用 `cos/sin` 而非 `cmath.exp`**:等价且避免复数构造开销,仅依赖 `math`。
- **holdover 默认关闭**:新状态机全部参数带默认值,默认配置下逐位复现旧行为,
  旧接口与回归测试无需修改。
- **冻结中位数而非均值**:中位数对进入缺口前的野值更稳健;窗口只记录有效样本的
  频率字,holdover 期间不更新(噪声不进入积分项)。
- **重锁定需双条件同时连续达标**:仅相位误差小可能是巧合(如频率错误的信号
  相位周期性扫过零点),频率误差条件排除这类假锁定。

## 已知限制

- 仅实现题面契约的二阶环路(无环路滤波器变体、无噪声建模、无辅助捕获/扫频)。
- `wmax` 不限制在 `pi` 以内;但采样系统频偏超过 `pi` rad/sample 会混叠,环路无法捕获,这属于信号条件而非缺陷。
- 不宣称兼容任何行业标准的完整 PLL 规范。
- 诊断全量保存在内存中,超长流需调用方自行分块并丢弃历史诊断。
- holdover 是纯开环外推:保持期间频率若真实漂移,外推相位误差会随时间累积,
  模块只报告该误差,不做任何"仍知真值"的承诺。

## 运行

```bash
python -m unittest discover -s tests -v   # 单元测试(30 项)
python demo.py                            # 捕获过程演示 + 非法输入拒绝, 约 0.1 秒
python demo_iteration.py                  # holdover/reacquire 状态机演示, 约 0.1 秒
```

## venv 准备

仅标准库,无需安装任何依赖;可用 venv 或系统 Python 3.14.7 直接运行:

```bash
python3.14 -m venv .venv
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python demo_iteration.py
```
