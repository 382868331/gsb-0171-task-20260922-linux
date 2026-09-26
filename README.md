# 整数采样数字锁相环(PLL)捕获诊断与导频丢失保持

基于合成复数导频的二阶离散 PLL 捕获模块,提供逐样本诊断、失锁冻结、
导频短暂消失时的 holdover/reacquire 状态机、任意分块一致处理与严格的
输入校验。

## 环境

- Python 3.14.7,Linux 原生运行
- 仅使用标准库(`math` / `dataclasses` / `unittest` / `json`),**无需安装任何第三方依赖**
- 核心、测试、演示全部离线;示例数据由仓库内确定性代码生成

### 准备 venv(可选)

不需要任何第三方包,系统 Python 即可直接运行;若希望隔离环境:

```bash
python3 -m venv .venv
.venv/bin/python -m unittest discover -s tests -v
```

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

## 导频丢失:tracking / holdover / reacquire 状态机

默认(`holdover_enter=None`)行为与上述基础契约逐位一致:低幅样本完全冻结。
设置 `holdover_enter = N` 后启用三态状态机(`Diagnostic.state` 可见):

- **tracking**:正常跟踪(基础契约行为)。低幅样本未达 N 个时也保持该状态并冻结。
- **holdover**:幅度连续 `< amin` 达到 `holdover_enter` 个样本后进入。
  冻结频率为最近 `holdover_window` 个有效 `omega` 的**中位数**(偶数个取中间两值均值),
  每个样本仅按该冻结频率推进相位(`theta' = wrap(theta + omega_hold)`),
  **不再用低幅噪声更新积分项**。低幅计数与所有状态在多次 `run()`/`step()`
  调用之间延续;中途出现一个有效样本会把连续低幅计数清零。
- **reacquire**:参考重新有效后进入。**不瞬间重置相位**:每样本相位校正项
  被限幅在 `±reacquire_step` 内(`theta' = wrap(theta + omega' + clip(kp*error, ±reacquire_step))`),逐渐追上参考。连续 `reacquire_count` 个样本**同时**满足
  `|error| <= eps` 且 `|omega - omega_hold| <= freq_eps` 才回到 tracking;
  回到 tracking 后按常规 `streak >= lock_count` 规则重新标记 `locked`
  (reacquire 期间一律 `locked=False`)。reacquire 期间再次出现低幅样本
  (假恢复)**立即**回到 holdover,无需重新计满门限。

**门限相等时的取向**(契约固定,测试断言):

- 幅度:`amp == amin` 算**有效**(无效判定严格为 `amp < amin`);
- 相位误差:`|error| == eps` 计入捕获/重捕获条件(`<=`);
- 频率误差:`|omega - omega_hold| == freq_eps` 计入重捕获条件(`<=`);
- 校正限幅:`|kp*error| == reacquire_step` 不额外削减(仅在严格超出时截断)。

**保持期间的误差口径**:参考丢失后 PLL 无法获知真实相位,只能按冻结频率外推;
若真实频率在缺口期间漂移,外推相位与真实相位的误差会随之变化(通常增长)。
`demo_iteration.py` 与 `tests/test_pll_holdover.py` 用已知真值的合成信号
**报告**该误差,而不声称保持期间仍能跟踪真实相位。

## 接口

```python
from pll_capture import PLL, PLLParams, PLLResult, Diagnostic

params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03,
                   lock_count=40,
                   holdover_enter=8,      # 可选; 默认 None 禁用 holdover
                   holdover_window=32, reacquire_step=0.08,
                   freq_eps=0.06, reacquire_count=30)
pll = PLL(params, theta0=0.0, omega0=0.0)   # |omega0| > wmax 会被拒绝而非静默限幅
diag = pll.step(z)                           # 单样本 -> Diagnostic
result = pll.run(samples)                    # 批量 -> PLLResult(含本批逐样本诊断)
result.omega                                 # 估计频偏 rad/sample
result.frequency_offset(48000.0)             # 换算 Hz: omega * fs / (2*pi)
diag.state                                   # "tracking" / "holdover" / "reacquire"
```

- `PLLParams(kp, ki, wmax, amin, eps, lock_count, holdover_enter=None, holdover_window=16, reacquire_step=0.05, freq_eps=0.01, reacquire_count=20)`:
  构造时校验,`int` 数值统一规范为 `float`;新增参数均有默认值,旧调用方式不变。
- `PLL(params, theta0=0.0, omega0=0.0)`:`theta0` 任意有限实数(内部 wrap 到 `[-pi, pi)`)。
- `Diagnostic`:`index, valid, amplitude, error, omega, theta, streak, locked, saturated, state`
  (新增 `state`,默认 `"tracking"`,旧的位置构造不受影响)。
- `PLLResult`:`params, omega, theta, locked, streak, lost_samples, processed, diagnostics, state`(新增 `state`)。
- 同一实例可任意分块连续调用 `run()`/`step()`,与一次性处理**逐位一致**(浮点运算顺序相同)。
- 模块常量 `STATE_TRACKING / STATE_HOLDOVER / STATE_REACQUIRE` 为状态名字符串。

## 错误语义

所有错误均为 `ValueError` 子类,消息中带参数名/样本值,可定位:

- `PLLParamError`:环路参数或初始状态非法(如 `kp >= 1`、`ki >= kp`、`lock_count` 非 int、
  `|omega0| > wmax`、`holdover_enter`/`reacquire_count` 非 int、`reacquire_step <= 0` 等)。
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
- **holdover 默认关闭**:`holdover_enter=None` 时行为与基础契约逐位一致,
  旧接口与回归测试全部保留;启用才改变低幅样本的处理方式。
- **冻结频率取中位数而非末值**:对进入缺口前的个别野值估计更稳健;
  中位数在确定性序列上可精确复现,分块一致性不受影响。
- **重捕获限幅只作用于校正项**:频率前馈 `omega'` 照常通过,否则大频偏下
  每样本总步长被限幅会永远追不上参考;限幅的是"追上"的速度,不是跟踪本身。
- **假恢复立即回 holdover**:reacquire 说明参考刚丢过,短暂有效后再跌落时
  冻结中位数仍是最近可用估计,重新计满门限只会白白用噪声污染积分项。

## 已知限制

- 仅实现题面契约的二阶环路(无环路滤波器变体、无噪声建模、无辅助捕获/扫频)。
- `wmax` 不限制在 `pi` 以内;但采样系统频偏超过 `pi` rad/sample 会混叠,环路无法捕获,这属于信号条件而非缺陷。
- 不宣称兼容任何行业标准的完整 PLL 规范。
- 诊断全量保存在内存中,超长流需调用方自行分块并丢弃历史诊断。
- holdover 是纯外推:缺口期间真实频率若漂移,重新锁定所需时间随之变长;
  漂移使 `|omega - omega_hold|` 持续超过 `freq_eps` 时会一直停留在 reacquire,
  需按场景调大 `freq_eps`。

## 运行

```bash
python -m unittest discover -s tests -v   # 单元测试(30 项: 19 项基础 + 11 项状态机)
python demo.py                            # 捕获过程演示 + 非法输入拒绝, 约 0.1 秒
python demo_iteration.py                  # holdover/reacquire 演示 + 门限相等边界, 约 0.1 秒
```
