# 整数采样数字锁相环(PLL)捕获诊断

基于合成复数导频的二阶离散 PLL 捕获模块,提供逐样本诊断、失锁冻结、
任意分块一致处理与严格的输入校验;可选启用周跳候选检测(确认后清零
锁定计数、保留频率估计,按调用方选择仅报告或按圆均值补偿相位)。

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

## 周跳候选检测(可选)

构造 `PLL` 时传入 `slip_config=CycleSlipConfig(...)` 启用;不传则完全关闭,
环路行为与未启用时逐位一致。

- **候选与确认**:有效样本 `|error|` 连续 `confirm_count`(K)次超过 `threshold`
  确认一次事件。事件保留**起始样本号**(`start_index`)、**确认号**(`confirm_index`)
  与**累计有符号误差**(`error_sum`,按 `[-pi, pi)` 原样累加)。
- **低幅度打断**:无效样本(`|z| < amin`)打断候选——不累计、不确认,
  也不伪造有效相位(诊断中 `error` 仍为 `None`);误差回到门限内的有效样本同样打断候选。
- **确认后的处理**:锁定计数(`streak`)清零、**频率估计 `omega` 保留**;
  按 `compensate` 选择仅报告,或在**该样本原更新之后**把相位补偿候选误差的
  **圆均值**(`theta += circular_mean`)。
- **退化方向**:候选误差单位向量的平均合向量长度 `resultant` 低于显式参数
  `deg_threshold` 时判定 `ambiguous=True`:只报告,`circular_mean=None`,
  不补偿、不随便挑方向。
- **防止连发**:确认候选的那个样本不会再次开启新候选;下一个候选最早从下一个样本开始。
- **跨 chunk 一致**:候选状态保存在实例内,任意分块处理与一次性处理逐位一致。

`CycleSlipConfig` 参数(构造时校验,非法抛 `PLLParamError`):

| 参数 | 含义 | 约束 |
| --- | --- | --- |
| `threshold` | 候选误差门限 (rad) | `0 < threshold <= pi` |
| `confirm_count` | 确认所需连续超限次数 K | `int >= 1`(拒绝 bool) |
| `deg_threshold` | 圆均值退化阈值(合向量长度) | `0 <= deg_threshold <= 1` |
| `compensate` | 确认后是否补偿相位 | 必须显式 `bool` |

`CycleSlipEvent` 字段:`start_index, confirm_index, count, error_sum,
circular_mean, resultant, ambiguous, compensated, correction`。

## 接口

```python
from pll_capture import PLL, PLLParams, PLLResult, Diagnostic, CycleSlipConfig

params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03, lock_count=40)
slip = CycleSlipConfig(threshold=0.5, confirm_count=3,
                       deg_threshold=0.3, compensate=True)
pll = PLL(params, theta0=0.0, omega0=0.0, slip_config=slip)  # slip_config 可省略
diag = pll.step(z)                           # 单样本 -> Diagnostic
result = pll.run(samples)                    # 批量 -> PLLResult(含本批逐样本诊断)
result.omega                                 # 估计频偏 rad/sample
result.frequency_offset(48000.0)             # 换算 Hz: omega * fs / (2*pi)
result.slip_events                           # 本批确认的周跳事件
pll.slip_events                              # 实例生命周期内累计事件
```

- `PLLParams(kp, ki, wmax, amin, eps, lock_count)`:构造时校验,`int` 数值统一规范为 `float`。
- `PLL(params, theta0=0.0, omega0=0.0, slip_config=None)`:`theta0` 任意有限实数(内部 wrap 到 `[-pi, pi)`)。
- `Diagnostic`:`index, valid, amplitude, error, omega, theta, streak, locked, saturated, slip_event`。
  确认事件的样本上,`theta`/`streak` 记录的是补偿与清零之后的值。
- `PLLResult`:`params, omega, theta, locked, streak, lost_samples, processed, diagnostics, slip_events`。
- 同一实例可任意分块连续调用 `run()`/`step()`,与一次性处理**逐位一致**(浮点运算顺序相同)。

## 错误语义

所有错误均为 `ValueError` 子类,消息中带参数名/样本值,可定位:

- `PLLParamError`:环路参数、周跳配置或初始状态非法(如 `kp >= 1`、`ki >= kp`、
  `lock_count` 非 int、`|omega0| > wmax`、`threshold > pi`、`compensate` 非 bool)。
- `PLLInputError`:样本非法(bool、非数值、NaN/Infinity)、批次不可迭代、单批超上限。

校验规则:数值输入拒绝 NaN/Infinity;整数参数(`lock_count`、`confirm_count`)拒绝
bool 与非 int;样本拒绝 bool;`compensate` 必须是显式 bool。非法输入一律抛错,**不静默纠正**。

**批次原子性**:`run()` 先整批校验再逐样本更新;任一样本非法时抛 `PLLInputError`,
PLL 状态(含周跳候选状态)保持调用前原样,不留部分变更。

## 输入上限

- `run()` 单批最多 `MAX_SAMPLES = 1_000_000` 个样本(诊断逐样本保存,限制内存;可分块处理更长的流)。
- 诊断内存占用为 O(本批样本数)。

## 误差口径(浮点)

- 所有计算为 IEEE 754 双精度;同一实例分块与整批处理结果**逐位相等**(测试以 `==` 断言)。
- 参考测试(`tests/data/*.json`)的期望值是导频的理论真值(生成相位/频偏),不是被测核心的输出:
  - 恒相位参考:收敛后 `|omega| <= 1e-9`、`|wrap(theta - 0.6)| <= 1e-9`、末端 `|error| <= 1e-9`;
  - 固定频偏参考:收敛后 `|omega - 0.3| <= 1e-9`、末端 `|error| <= 1e-6`。
- 周跳测试的触发位置(起始号/确认号)由预设误差序列手工数出;圆均值/合向量长度
  在测试内用 `math` 按定义独立重算对照。
- 收敛为指数渐近,上述容差对应数百个样本后的稳态残差量级,不用于掩盖边界行为
  (限幅、冻结、计数清零、候选打断均以精确条件断言)。

## 设计取舍

- **先校验后更新**:`run()` 牺牲一次遍历换取批次原子性,避免有状态对象的部分变更。
- **无效样本清零捕获计数**:契约要求"连续 L 个有效样本",无效样本打断连续性,故同时退出 locked。
- **`omega0` 超限拒绝而非限幅**:初始状态属于调用方显式输入,静默修改会掩盖配置错误。
- **旋转用 `cos/sin` 而非 `cmath.exp`**:等价且避免复数构造开销,仅依赖 `math`。
- **补偿用圆均值而非算术均值**:候选误差可能分布在 ±pi 两侧(如导频反转),
  算术均值会给出错误方向;圆均值对单位向量求平均,跨界时方向正确。
- **`error_sum` 按原样累加**:它是记录字段,跨 ±pi 时不代表真实偏移;补偿从不使用它。
- **确认样本不再开启新候选**:该样本的误差已计入刚确认的事件,立即复用会导致连发。

## 已知限制

- 仅实现题面契约的二阶环路(无环路滤波器变体、无噪声建模、无辅助捕获/扫频)。
- `wmax` 不限制在 `pi` 以内;但采样系统频偏超过 `pi` rad/sample 会混叠,环路无法捕获,这属于信号条件而非缺陷。
- 周跳检测是对**合成导频离线数据**的启发式判决,基于误差门限与计数,
  **不构成真实硬件诊断保证**;门限/K/退化阈值需按信号条件整定。
- 候选误差是环路跟踪中的衰减误差,确认时环路已跟踪掉一部分偏移,
  因此按圆均值补偿可能超过剩余偏移量(过冲),随后由环路自身收敛吸收。
- 捕获暂态(未锁定前的大误差)同样会触发候选/事件,需先用 `theta0`/`omega0`
  初始化或忽略锁定前的事件。
- 不宣称兼容任何行业标准的完整 PLL 规范。
- 诊断全量保存在内存中,超长流需调用方自行分块并丢弃历史诊断。

## 运行

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 单元测试(36 项)
./.venv/bin/python demo.py                            # 捕获 + 非法输入拒绝 + 周跳检测演示, 约 0.1 秒
```
