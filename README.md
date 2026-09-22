# 整数采样数字锁相环（DPLL）捕获诊断模块

离线通信同步工程用的可复用小库：对**合成复数导频**运行二阶离散锁相环，
输出频偏/相位估计、锁定状态与逐样本诊断（相位误差、限幅、失锁冻结、
连续合格计数）。纯 Python 标准库实现，不调用任何第三方 PLL/DSP 实现。

- **Python 版本**：CPython 3.14.7（使用 `math`，不依赖版本新特性，3.10+ 语法层面亦可运行，但验收以 3.14.7 为准）
- **依赖**：无。标准库之外不安装任何包；不需要网络、账号、硬件、数据库或 Docker
- **平台**：Linux 原生

## 目录

```
pll_diag/          本题库（可导入包）
  __init__.py       公开接口导出
  model.py          PLLParams / PLLState / PLLSample / PLLDiagnosis（frozen dataclass）
  core.py           PLL 状态机、track() 便捷接口、校验与错误类型
tests/
  reference.py      独立参考实现（oracle），不 import 被测核心
  test_tracking.py  跟踪正确性：独立参考逐样本比对 + 解析参考 + 边界场景
  test_validation.py 参数/输入校验、原子性、接口边界
demo.py            固定输入离线演示（确定性合成导频，无 sleep、无预录结果）
```

## 运行命令（准确）

在本目录（包含 `pll_diag/`、`tests/`、`demo.py` 的目录）下：

```bash
python -m unittest discover -s tests -v   # 执行全部必要测试
python demo.py                             # 约 2 ms 完成（远小于 8 秒预算）
```

## 算法契约

对复数样本 `z`、当前相位估计 `theta`、频偏估计 `omega`，每步：

```
r     = z * exp(-j*theta)                  # 先按 theta 旋转样本
error = atan2(imag r, real r) ∈ [-π, π)   # atan2 的 +π 归入 -π
omega_unc = omega + ki * error
omega' = clip(omega_unc, -wmax, +wmax)
theta' = wrap(theta + omega' + kp*error) ∈ [-π, π)
```

- 幅度 `abs(z) < amin`（严格小于）：**冻结** `theta/omega`，该样本计失锁，
  连续合格计数清零，`error=None`。
- 有效样本：`abs(error) <= eps`（等号含）则连续计数 +1，否则**立即清零**；
  计数 `>= L` 起 `locked=True`。
- 参数在任何处理前校验稳定范围：`0 < kp < 1`、`0 < ki < kp`。

角度单位：相位为弧度，频偏为弧度/样本（“整数采样”指每个输入样本推进一拍，
不做样本率换算）。

## 公开接口

```python
from pll_diag import PLL, PLLParams, PLLState, PLLSample, PLLDiagnosis, track
from pll_diag import PLLConfigError, PLLInputError

params = PLLParams(kp=0.3, ki=0.05, wmax=0.5, eps=2e-2, L=8, amin=1e-12)

# 一次性
diag = track(samples, params)                 # samples: Iterable[complex|int|float]

# 任意分块（状态自动延续）
pll = PLL(params)
d1 = pll.update(block1)
d2 = pll.update(block2)                        # pll.state == d2.final_state
```

- `PLLParams`：`kp, ki, wmax, eps, L, amin=1e-12`（不可变）。
- `PLLState`：`theta, omega, streak, locked, processed, lost_count`（不可变）。
  可用 `PLL(params, state)` 从已知状态续跑，或 `pll.reset()`。
- `PLLSample`：`index, amplitude, valid, error, theta, omega,
  omega_saturated, streak_after, locked_after`。
  **`index` 是全局样本下标**（等于处理该样本前的 `processed`），因此任意
  分块得到的逐样本诊断与一次性处理逐字段相同。
- `PLLDiagnosis`：`samples, final_state, locked, locked_at, lost_indices,
  saturated_indices, max_abs_error`，便捷属性 `frequency_offset`、`phase`、
  `status`（`"locked"/"unlocked"`）。`locked_at/lost_indices/...` 均为
  全局下标；每次 `update` 只报告本段内发生的事件。
- 空序列是合法输入：返回空诊断、状态不变。

## 输入上限（接口边界）

| 边界 | 值 | 违反时 |
|---|---|---|
| 单次 `update`/`track` 样本数 | `MAX_SAMPLES_PER_CALL = 10_000_000` | `PLLInputError` |
| 样本实部/虚部绝对值 | `MAX_COMPONENT_MAGNITUDE = 1e100` | `PLLInputError` |
| 连续锁定阈值 `L` | `1 <= L <= MAX_L = 2_147_483_647` | `PLLConfigError` |

常量可从 `pll_diag.core` 导入。上限是防护性接口边界，不是验收规模；
演示与测试只使用小数据。

## 错误语义

- `PLLConfigError(ValueError)`：参数不合法，错误信息包含参数名与取值，按
  固定顺序（有限性 → `L` 正整数 → `0<kp<1` → `0<ki<kp` → 正数项）抛出。
- `PLLInputError(ValueError)`：样本/初始状态不合法，信息带可定位下标
  （如 `samples[3] 拒绝 NaN`）。
- 所有数值输入拒绝 `NaN/Infinity`；`bool` 虽是 `int` 子类，整数参数与
  布尔样本一律显式拒绝（`L=True` 不接受）。
- 非数值样本（`str`、`None`、列表等）拒绝；`samples` 不可迭代时拒绝。
- **原子性**：`update` 先整段物化并校验全部样本，再改状态；任一样本非法
  或超长，整段拒绝、`pll.state` 保持调用前快照，无部分变更。`reset()`
  对非法状态同样保持现状。
- 初始状态校验：`theta ∈ [-π,π)`、`omega ∈ [-wmax,wmax]`、计数非负整数，
  且 `locked == (streak >= L)` 必须自洽。

本模块不存在“搜索预算耗尽 vs 证明无解”这类搜索算法场景；PLL 是逐样本
确定性递推，输入有限必在 O(N) 内给出结果，无“无解”输出。

## 浮点结果与误差口径

- 全部用 IEEE-754 double（CPython `float`）；旋转用 `cos/sin` 展开的
  实部/虚部分式，回绕用 `% (2π)`，限幅用 `max/min`。
- 测试中的**独立参考**（`tests/reference.py`）用不同运算次序
  （`cmath.exp` 旋转、`math.fmod` 回绕、显式比较限幅）独立编写，不 import
  被测核心；两边逐样本比对口径为误差/相位/频偏 `places=12`（约 1e-12），
  布尔/计数字段精确相等。
- 对解析真值（恒相位、固定频偏导频）使用**紧口径 1e-9**：实测收敛后
  `|omega - dphi|`、相位差均在 1e-15 量级（见 demo 打印），不用宽容差
  掩盖边界错误。
- 相位约定：处理完样本 k 的 `theta` 是更新式中对**下一个**样本（k+1）
  相位的预测（`theta' = theta + omega'`），比较真值时按此口径。
- `amin` 判定为严格小于；`eps` 与 `±wmax` 边界为含等号。

## 设计取舍

- **二阶 PI 结构、`kp` 作用于相位更新、`ki` 作用于积分器**：严格按题面
  更新式实现，不自选行业常见的其他归一化形式。
- **全局样本下标**：牺牲“每块从 0 编号”的直觉，换取任意分块与一次性
  处理的逐样本诊断严格相等（测试覆盖 cut=1 起的多种分块、含空块）。
- **失锁即冻结且清零计数**：不对幅度做归一化/插值，失锁期间不产生
  `error`（`None`），不把猜测当观测。
- **frozen dataclass + slots**：诊断结果不可变，可安全在分块间传递比较。
- 先校验后处理、整段物化：用 O(块长) 临时内存换取失败原子性。

## 已知限制（只实现题面契约的数学子集）

- 不实现鉴频器、锁定检测的其他统计量（方差/置信度）、环路带宽自动设计、
  符号定时恢复、 Gardner/TED、Costas 半波判决等；不宣称与任何通信行业
  标准兼容。
- 不做噪声蒙特卡洛分析；`eps/L` 是给定的确定性锁定规则，不保证信噪比
  意义下的虚警/漏警概率。
- 输入被当作单位时刻整数采样序列，不接受采样率/时间戳，不换算 Hz。
- `theta` 与频偏估计是绕回/限幅后的主值，不维护连续解卷绕相位
  （需要时可由逐样本 `error` 自行累加）。
- 大输入为同步单线程处理，无并行/流式增量之外的内存优化
  （生成器可传入，但当前会物化为列表以保证原子校验）。
