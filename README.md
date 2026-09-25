# 整数采样数字锁相环(PLL)捕获诊断

基于合成复数导频的二阶离散 PLL 捕获模块,提供逐样本诊断、失锁冻结、
任意分块一致处理、严格的输入校验,以及**可选的周跳候选检测**。

## 环境

- Python 3.14.7,Linux 原生运行
- 仅使用标准库(`math` / `dataclasses` / `typing` / `unittest` / `json`),
  **无需安装任何第三方依赖**
- 核心、测试、演示全部离线;示例数据由仓库内确定性代码生成
- 推荐通过仓库自带 venv 运行:

```bash
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python demo.py
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

## 可选周跳候选检测

默认**关闭**,旧代码行为与逐位结果完全不变。启用方式是在 `PLLParams` 中同时
给出 `jump_k=K` 与 `jump_eps`(二者必须同时给出或同时为 `None`):

- 有效样本满足严格 `|error| > jump_eps` 即计入候选,候选保留**起始样本号**与
  **累计有符号误差**(`error_sum`,K 个 `[-pi, pi)` 误差的算术和)。
- 连续 **K** 个越限有效样本确认一次事件 `JumpCandidate`;事件含
  `start_index`、`confirm_index`、`error_sum`、K 个原始误差 `errors`、
  圆均值 `circular_mean` 与合向量长度 `r`。
- **打断而非确认**:候选期间出现低幅度无效样本,或 `|error| <= jump_eps` 的
  有效样本,候选作废,不产生事件;无效样本依旧冻结、`error=None`,
  **不伪造有效相位**。
- **确认时**:锁定计数强制清零(确认样本 `streak=0`),频率估计 `omega` 原样
  保留(不重置、不限幅之外的改动)。
- **仅报告 vs 补偿**由构造参数 `jump_compensate` 选择:
  - `False`(默认):只在诊断/结果里报告事件,`theta`、`omega` 与关闭检测时
    **逐位一致**;
  - `True`:在确认样本的**原更新完成之后**,把相位再叠加候选误差的圆均值
    `theta = wrap(theta + circular_mean)`;补偿不改变 `omega`。同一样本不会
    因补偿后的状态再次开启候选,不连发。
- **圆均值与退化阈值**:`r = |Σ exp(j·e_k)| / K ∈ [0,1]`,
  `circular_mean = wrap(atan2(Σsin, Σcos))`,对跨 ±pi 的误差簇取最短弧方向。
  当 `r < r_min`(显式参数,`0 <= r_min <= 1`,默认 `0.5`)时合向量退化,
  事件标记 `ambiguous=True`、`circular_mean=None`,补偿模式下也**不挑方向、
  不改 theta**。判定为严格小于,恰等阈值不算退化。

> 说明:周跳检测只是对合成相位误差序列的启发式标注。它不是、也不应被当作
> 真实硬件的周跳/失锁诊断结论或保证;捕获瞬态、长滑脱等同样可能满足 K 连越
> 限条件,调用方需结合信号条件自行解释。

## 接口

```python
from pll_capture import PLL, PLLParams, PLLResult, Diagnostic, JumpCandidate, circular_rmean

# 不传 jump_k/jump_eps:检测关闭,与旧版完全一致
params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03, lock_count=40)

# 启用检测: K=3, 越限门限 0.3, 圆均值退化阈值 r_min=0.5
params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03,
                   lock_count=40, jump_k=3, jump_eps=0.3, r_min=0.5)

pll = PLL(params, theta0=0.0, omega0=0.0, jump_compensate=False)  # True=确认后补偿
diag = pll.step(z)          # 单样本 -> Diagnostic
result = pll.run(samples)   # 批量 -> PLLResult
result.jumps                # 本批确认的 JumpCandidate(仅本批)
result.jump_count           # 实例生命周期内累计事件数
pll.candidate_active        # 是否存在跨样本(可跨批)未确认候选

mean, r = circular_rmean([3.05, 3.12, -3.08], r_min=0.5)  # 独立工具函数
```

- `PLLParams(..., jump_k=None, jump_eps=None, r_min=0.5)`:检测禁用时
  `r_min` 仍做取值校验;`int` 数值统一规范为 `float`。
- `Diagnostic` 在原 9 个字段后追加(均有默认值,旧位置构造不受影响):
  `candidate`(本样本之后候选是否挂起)、`jump`(本样本确认的
  `JumpCandidate`,否则 `None`)。
- `PLLResult` 追加 `jumps`(本批事件元组)与 `jump_count`(累计数),
  同样带默认值。
- 同一实例可任意分块连续调用 `run()`/`step()`:环路状态与候选状态都跨调用
  保留,任意切块(含候选恰好跨在边界上)与一次性处理**逐位一致**。
- 批次事件只在确认样本所在的那一批返回,不跨批重复携带。

## 错误语义

所有错误均为 `ValueError` 子类,消息中带参数名/样本值,可定位:

- `PLLParamError`:环路参数或初始状态非法(如 `kp >= 1`、`ki >= kp`、
  `lock_count` 非 int、`|omega0| > wmax`),或检测配置非法:`jump_k`/`jump_eps`
  只给出一个、`jump_k < 1` 或非 int、`jump_eps <= 0`、`r_min` 不在
  `[0, 1]`、`jump_compensate` 非 bool。
- `PLLInputError`:样本非法(bool、非数值、NaN/Infinity)、批次不可迭代、
  单批超上限;`circular_rmean` 的角度序列为空或含非有限值。

校验规则:数值输入拒绝 NaN/Infinity;整数参数(`lock_count`、`jump_k`)
拒绝 bool 与非 int;样本拒绝 bool。非法输入一律抛错,**不静默纠正**。

**批次原子性**:`run()` 先整批校验再逐样本更新;任一样本非法时抛 `PLLInputError`,
PLL 状态(含候选状态)保持调用前原样,不留部分变更。

## 输入上限

- `run()` 单批最多 `MAX_SAMPLES = 1_000_000` 个样本(诊断逐样本保存,限制内存;可分块处理更长的流)。
- 诊断内存占用为 O(本批样本数);每个确认事件保留 K 个误差,无额外流式内存。

## 误差口径(浮点)

- 所有计算为 IEEE 754 双精度;同一实例分块与整批处理结果**逐位相等**(测试以 `==` 断言,含 `JumpCandidate`)。
- 参考测试(`tests/data/*.json`)的期望值是导频的理论真值(生成相位/频偏),不是被测核心的输出:
  - 恒相位参考:收敛后 `|omega| <= 1e-9`、`|wrap(theta - 0.6)| <= 1e-9`、末端 `|error| <= 1e-9`;
  - 固定频偏参考:收敛后 `|omega - 0.3| <= 1e-9`、末端 `|error| <= 1e-6`。
- 周跳相关断言以**精确触发位置**(`start_index`/`confirm_index` 相等)、符号与
  独立重算的圆均值/`r` 给出,不用宽容差掩盖边界行为;收敛为指数渐近,
  稳态容差仅用于收敛后的残差量级。

## 设计取舍

- **检测严格 opt-in**:`jump_k`/`jump_eps` 默认 `None`,不改变任何既有结果;
  新增 dataclass 字段全部带默认值,旧的位置构造可用。
- **先校验后更新**:`run()` 牺牲一次遍历换取批次原子性,避免有状态对象的部分变更。
- **确认只清零计数、保留频率**:题面要求频率估计保留;事件后是否重新锁定由后续
  真实误差自然决定,不由检测器伪造。
- **补偿用圆均值而非算术和**:K 个误差各自 wrap 在 `[-pi, pi)`,跨边界时算术和
  会指错方向(如 `+3.05, +3.12, -3.08` 算术和为 +3.09 而真实方向在 +pi 附近);
  圆均值取最短弧,退化时显式报 `ambiguous` 而非任选方向。
- **补偿在原更新之后且当步不重入**:避免补偿量被同一候选重复计入造成连发。
- **`omega0` 超限拒绝而非限幅**:初始状态属于调用方显式输入,静默修改会掩盖配置错误。
- **旋转用 `cos/sin` 而非 `cmath.exp`**:等价且避免复数构造开销,仅依赖 `math`。

## 已知限制

- 仅实现题面契约的二阶环路(无环路滤波器变体、无噪声建模、无辅助捕获/扫频)。
- `wmax` 不限制在 `pi` 以内;但采样系统频偏超过 `pi` rad/sample 会混叠,环路无法捕获,这属于信号条件而非缺陷。
- 检测器是**门限+计数的启发式**:门限取值、K 与 `r_min` 需调用方按信号条件选择;
  捕获瞬态或持续滑脱也可能确认事件,`ambiguous` 只表示候选内部方向不一致,
  非 `ambiguous` 也不等于"真实发生了硬件周跳"。本模块不对真实硬件做任何诊断保证。
- 补偿为单次开环相位步进,不保证补偿后立即锁定(仍需后续样本重新满足 L 连小门限)。
- 不宣称兼容任何行业标准的完整 PLL 规范。
- 诊断全量保存在内存中,超长流需调用方自行分块并丢弃历史诊断。

## 运行

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 单元测试(45 项: 19 旧回归 + 26 新增)
./.venv/bin/python demo.py                            # 捕获/周跳检测/补偿/拒绝演示, 约 0.1 秒
```

演示覆盖:正常捕获与分块一致、孤立毛刺(打断不确认)、跨 ±pi 圆均值、
真实阶跃(清零锁定/保留频率/重新锁定,仅报告与补偿两种模式)、低幅度中断、
合向量退化 `ambiguous`、候选跨切块边界,以及参数/输入拒绝。
