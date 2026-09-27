# 整数采样数字锁相环(PLL)捕获诊断

基于合成复数导频的二阶离散 PLL 捕获模块,提供逐样本诊断、失锁冻结、
任意分块一致处理与严格的输入校验;并提供**双参考无扰动切换交接器**:
新参考先独立跟踪,确认稳定后再无相位跳变地交接,切换窗口内原参考失效时
明确进入不可信的保持态(holdover),来回切换不累积人为相位偏移。

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

## 接口

```python
from pll_capture import PLL, PLLParams, PLLResult, Diagnostic

params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03, lock_count=40)
pll = PLL(params, theta0=0.0, omega0=0.0)   # |omega0| > wmax 会被拒绝而非静默限幅
diag = pll.step(z)                           # 单样本 -> Diagnostic
result = pll.run(samples)                    # 批量 -> PLLResult(含本批逐样本诊断)
result.omega                                 # 估计频偏 rad/sample
result.frequency_offset(48000.0)             # 换算 Hz: omega * fs / (2*pi)
```

### 双参考无扰动交接

两路参考**采样速率相同**,每步成对喂入 `(z0, z1)`;两路各挂一个独立的
`PLL`,始终同时、独立更新。输出相位不是某一路的裸相位,而是
`wrap(theta_i + offset_i)`:

```python
from pll_capture import DualReferencePLL, MODE_TRACKING, MODE_HOLDOVER

d = DualReferencePLL(params)                 # 两路共用同一组 PLLParams
d.run([(z0_k, z1_k) for k in range(N)])      # 成对样本;首个锁定的参考先接管
d.request_switch(1)                          # 请求切到参考 1(不立即跳变)
result = d.run(pairs)                        # 目标稳定(其 PLL locked)后自动交接
d.selected        # 当前输出依据的参考: 0/1;无任何稳定参考时为 None
d.mode            # tracking(可信) / acquiring(从未锁定) / holdover(失效保持)
d.trusted         # 仅 tracking 为 True
d.theta, d.omega # 输出相位/频偏;holdover 时冻结且 trusted=False
d.offset(0)       # 该参考当前的输出对齐偏置(rad, [-pi, pi))
d.cancel_switch() # 撤销尚未完成的切换请求
```

交接语义:

- **目标参考先独立跟踪**:调用 `request_switch()` 后,目标参考继续独立收敛,
  必须连续 `lock_count` 个有效样本误差在 `eps` 内(其 PLL 进入 locked)才确认稳定;
  未稳定前输出继续依据原参考,诊断中 `requested=目标`、`active=原参考`。
- **交接样本零跳变**:交接瞬间按两路绝对裸相位**重新计算一次性偏置**
  `offset_new = wrap(theta_old_out - theta_new_raw)`,交接样本输出相位与
  上一样本严格相等;偏置不继承历史值,因此 0→1→0→1 来回切换不会反复累积
  人为相位偏移。系统首次接管参考时 `offset=0`,输出直接取该参考裸相位。
- **切换窗口原参考失效**:若原参考样本无效(幅度 `< amin`)而目标尚未稳定,
  进入 `holdover`:输出冻结在上一可信值,`mode='holdover'`、`trusted=False`,
  **不假装仍锁定**;切换请求保留,目标继续独立跟踪,一旦确认稳定即在该样本
  无扰动交接并恢复 `tracking`。自启动以来从未有参考锁定时为 `acquiring`。
- **无挂起请求时活动参考失效**:同样进入 `holdover`,等待活动参考重新出现
  并重新确认锁定后接管;不会未经调用方请求就静默改用另一路参考。
- 输入为 `(z0, z1)` 样本对(长度恰为 2 的 tuple),样本校验规则与单路一致;
  `run()` 先整批校验再更新,非法样本对抛 `PLLInputError` 且**两路 PLL 与
  交接器状态全部回滚**到调用前(批次原子性)。
- `DualRefDiagnostic` 字段:`index, active, mode, trusted, handoff, requested,
  selected, theta, omega, lock0, lock1, valid0, valid1, offset,
  holdover_samples`;`DualRefResult` 另含 `handoff_count` 等快照字段。

- `PLLParams(kp, ki, wmax, amin, eps, lock_count)`:构造时校验,`int` 数值统一规范为 `float`。
- `PLL(params, theta0=0.0, omega0=0.0)`:`theta0` 任意有限实数(内部 wrap 到 `[-pi, pi)`)。
- `Diagnostic`:`index, valid, amplitude, error, omega, theta, streak, locked, saturated`。
- `PLLResult`:`params, omega, theta, locked, streak, lost_samples, processed, diagnostics`。
- 同一实例可任意分块连续调用 `run()`/`step()`,与一次性处理**逐位一致**(浮点运算顺序相同)。

## 错误语义

所有错误均为 `ValueError` 子类,消息中带参数名/样本值,可定位:

- `PLLParamError`:环路参数或初始状态非法(如 `kp >= 1`、`ki >= kp`、`lock_count` 非 int、
  `|omega0| > wmax`)。
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
- 双参考交接样本的输出相位与上一样本**逐位相等**(测试以 `==` 断言),
  即交接不产生相位跳变;交接后输出随活动参考的 PLL 连续演化。
- 对齐偏置是**每次交接按两路绝对裸相位重新计算的一次性** `wrap` 结果,
  恒在 `[-pi, pi)`,不与历史偏置相加,故反复切换不累积人为偏移。偏置在交接
  瞬间固化,其中含目标环路锁定确认瞬间的残差(量级 `< eps`),属真实跟踪残差
  而非人为跳变,并随目标继续收敛指数减小。
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

## 已知限制

- 仅实现题面契约的二阶环路(无环路滤波器变体、无噪声建模、无辅助捕获/扫频)。
- `wmax` 不限制在 `pi` 以内;但采样系统频偏超过 `pi` rad/sample 会混叠,环路无法捕获,这属于信号条件而非缺陷。
- 不宣称兼容任何行业标准的完整 PLL 规范。
- 诊断全量保存在内存中,超长流需调用方自行分块并丢弃历史诊断。
- 双参考交接器要求两路采样速率相同、样本成对(同一时基逐样本对齐);不处理
  异步采样率转换或时间戳对齐。
- 交接偏置仅保证**相位**无跳变;两路参考存在真实频差时,输出频偏会在交接
  后跟随活动参考的 `omega`(频偏本身的切换不是跳变,而是环路的连续估计)。

## 运行

```bash
python -m unittest discover -s tests -v   # 单元测试(29 项)
python demo.py                            # 单路捕获 + 双参考无扰动切换 + 非法输入拒绝, 约 0.1 秒
```
