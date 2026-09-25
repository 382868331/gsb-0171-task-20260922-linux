"""整数采样数字锁相环(PLL)捕获诊断模块。

基于合成复数导频的二阶离散 PLL。每个有效样本按如下顺序更新:

1. 旋转:   y = z * exp(-j*theta)
2. 误差:   error = wrap(atan2(Im y, Re y))  ∈ [-pi, pi)
3. 频率:   omega' = clip(omega + ki*error, -wmax, +wmax)
4. 相位:   theta' = wrap(theta + omega' + kp*error)

幅度 < amin 的样本视为无效:冻结 omega/theta 更新、清零捕获计数、
计一次失锁并在诊断中标记 valid=False。

连续 lock_count 个有效样本满足 |error| <= eps 时进入 locked;
任一有效样本 |error| > eps 立即清零计数并退出 locked。

可选周跳候选检测(jump_k/jump_eps 同时给出时启用, 均为 None 时禁用):

- 有效样本 |error| > jump_eps("超过"为严格大于)累计为候选; 候选保留起始
  样本号与累计有符号误差。连续 jump_k(K)个越限有效样本即确认一次事件,
  事件记录起始样本号、确认样本号、累计有符号误差以及该 K 个误差的圆均值
  和合向量长度 r。
- 候选期间出现低幅度样本(无效)或 |error| <= jump_eps 的有效样本则打断
  候选:不确认事件, 无效样本不伪造有效相位(仍走冻结路径)。
- 确认后:锁定计数清零, 频率估计 omega 原样保留。按 PLL 构造时
  jump_compensate 的选择:
  * False(默认, 仅报告):theta 维持原更新结果, 检测对环路无任何扰动;
  * True(补偿):在该样本原更新完成之后, 将 theta 再叠加候选误差的圆均值
    theta = wrap(theta + mean)。同一样本不再重新开启候选、不重复补偿。
- 圆均值合向量长度 r < r_min(显式参数)时事件标记 ambiguous, mean=None,
  补偿模式下也不挑方向、不改 theta。

注意:周跳检测只是对合成相位误差序列的启发式标注, 不构成真实硬件的
周跳/失锁诊断结论或保证。

仅依赖 Python 标准库(math / dataclasses)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "PLLParams",
    "PLL",
    "Diagnostic",
    "PLLResult",
    "JumpCandidate",
    "PLLError",
    "PLLParamError",
    "PLLInputError",
    "MAX_SAMPLES",
    "wrap_phase",
    "circular_rmean",
]

#: run() 单次调用允许的最大样本数(诊断逐样本保存,限制内存占用)。
MAX_SAMPLES = 1_000_000

_TWO_PI = 2.0 * math.pi


class PLLError(ValueError):
    """本模块所有参数/输入错误的基类(ValueError 子类)。"""


class PLLParamError(PLLError):
    """环路参数或初始状态非法。"""


class PLLInputError(PLLError):
    """输入样本非法。"""


def wrap_phase(angle: float) -> float:
    """把弧度相位规范到 [-pi, pi)。"""
    return (angle + math.pi) % _TWO_PI - math.pi


def circular_rmean(angles, r_min: float) -> Tuple[Optional[float], float]:
    """圆均值与合向量长度。

    把 angles 视为单位圆上的角度(rad), 返回 (mean, r):

    - r = |Σ exp(j·a)| / n ∈ [0, 1];
    - r >= r_min 时 mean = wrap(atan2(Σsin, Σcos)) ∈ [-pi, pi), 可正确处理
      跨 ±pi 边界的角度簇(如 3.05, 3.1, -3.12 的均值在 +pi 附近);
    - r < r_min 时合向量退化(方向不唯一), 返回 (None, r), 调用方不得据此
      任选方向。r_min=0 时永不判退化(恰为零向量时 atan2 给出 0.0, 由
      调用方自行承担);r_min=1 时仅所有角度完全同向才返回均值。

    angles 必须为非空有限实数序列;r_min 为有限实数且 0 <= r_min <= 1。
    """
    if isinstance(angles, (str, bytes)):
        raise PLLInputError("angles: 需要角度的可迭代序列, 拒绝字符串")
    try:
        items = list(angles)
    except TypeError:
        raise PLLInputError(
            f"angles: 需要可迭代序列, 得到 {type(angles).__name__}"
        ) from None
    if not items:
        raise PLLInputError("angles: 空序列无法定义圆均值")
    rmin = _finite_real("r_min", r_min, PLLParamError)
    if not 0.0 <= rmin <= 1.0:
        raise PLLParamError(f"r_min: 需要 0 <= r_min <= 1, 得到 {rmin!r}")
    sx = 0.0
    sy = 0.0
    for a in items:
        v = _finite_real("angles[]", a, PLLInputError)
        sx += math.cos(v)
        sy += math.sin(v)
    n = len(items)
    r = math.hypot(sx, sy) / n
    if r < rmin:
        return None, r
    return wrap_phase(math.atan2(sy, sx)), r


def _finite_real(name: str, value, exc) -> float:
    """校验有限实数;拒绝 bool、非数值类型、NaN 与 Infinity。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise exc(f"{name}: 需要有限实数, 得到 {type(value).__name__}({value!r})")
    v = float(value)
    if not math.isfinite(v):
        raise exc(f"{name}: 拒绝 NaN/Infinity, 得到 {value!r}")
    return v


def _coerce_sample(z) -> Tuple[float, float]:
    """校验单个样本并返回 (re, im)。实数视为虚部为 0 的复数。"""
    if isinstance(z, bool):
        raise PLLInputError(f"sample: 拒绝 bool 样本 {z!r}")
    if isinstance(z, complex):
        re, im = z.real, z.imag
    elif isinstance(z, (int, float)):
        re, im = float(z), 0.0
    else:
        raise PLLInputError(
            f"sample: 需要 complex/float/int, 得到 {type(z).__name__}({z!r})"
        )
    if not (math.isfinite(re) and math.isfinite(im)):
        raise PLLInputError(f"sample: 拒绝 NaN/Infinity 样本 {z!r}")
    return re, im


@dataclass(frozen=True)
class PLLParams:
    """环路参数。构造时校验稳定范围: 0 < kp < 1 且 0 < ki < kp。

    kp:         比例增益(相位步进),必须满足 0 < kp < 1。
    ki:         积分增益(频率步进),必须满足 0 < ki < kp。
    wmax:       频率字限幅幅度(rad/sample),有限且 > 0。
    amin:       有效样本最小幅度,有限且 >= 0。
    eps:        捕获误差门限(rad),有限且 > 0。
    lock_count: 进入 locked 所需的连续有效样本数 L,int 且 >= 1(拒绝 bool)。

    周跳候选检测(可选, 严格向后兼容):
    jump_k:     确认事件所需的连续越限有效样本数 K,int 且 >= 1;
                与 jump_eps 必须同时给出或同时为 None。
    jump_eps:   候选越限门限(rad),严格 |error| > jump_eps 才计数,有限且 > 0。
    r_min:      圆均值退化阈值,0 <= r_min <= 1;候选合向量 r < r_min 时
                事件标记 ambiguous 且不给出补偿方向。检测禁用时仍校验取值。
    """

    kp: float
    ki: float
    wmax: float
    amin: float
    eps: float
    lock_count: int
    jump_k: Optional[int] = None
    jump_eps: Optional[float] = None
    r_min: float = 0.5

    def __post_init__(self) -> None:
        kp = _finite_real("kp", self.kp, PLLParamError)
        ki = _finite_real("ki", self.ki, PLLParamError)
        wmax = _finite_real("wmax", self.wmax, PLLParamError)
        amin = _finite_real("amin", self.amin, PLLParamError)
        eps = _finite_real("eps", self.eps, PLLParamError)
        if not 0.0 < kp < 1.0:
            raise PLLParamError(f"kp: 需要 0 < kp < 1, 得到 {kp!r}")
        if not 0.0 < ki < kp:
            raise PLLParamError(f"ki: 需要 0 < ki < kp(={kp!r}), 得到 {ki!r}")
        if not wmax > 0.0:
            raise PLLParamError(f"wmax: 需要 > 0, 得到 {wmax!r}")
        if not amin >= 0.0:
            raise PLLParamError(f"amin: 需要 >= 0, 得到 {amin!r}")
        if not eps > 0.0:
            raise PLLParamError(f"eps: 需要 > 0, 得到 {eps!r}")
        lc = self.lock_count
        if isinstance(lc, bool) or not isinstance(lc, int):
            raise PLLParamError(
                f"lock_count: 需要 int(拒绝 bool), 得到 {type(lc).__name__}({lc!r})"
            )
        if lc < 1:
            raise PLLParamError(f"lock_count: 需要 >= 1, 得到 {lc!r}")

        jk = self.jump_k
        je = self.jump_eps
        if jk is None and je is None:
            jump_eps_norm = None
        else:
            if jk is None or je is None:
                raise PLLParamError(
                    "jump_k/jump_eps: 必须同时给出(启用检测)或同时为 None(禁用), "
                    f"得到 jump_k={jk!r}, jump_eps={je!r}"
                )
            if isinstance(jk, bool) or not isinstance(jk, int):
                raise PLLParamError(
                    f"jump_k: 需要 int(拒绝 bool), 得到 {type(jk).__name__}({jk!r})"
                )
            if jk < 1:
                raise PLLParamError(f"jump_k: 需要 >= 1, 得到 {jk!r}")
            jump_eps_norm = _finite_real("jump_eps", je, PLLParamError)
            if not jump_eps_norm > 0.0:
                raise PLLParamError(
                    f"jump_eps: 需要 > 0, 得到 {jump_eps_norm!r}"
                )
        rmin = _finite_real("r_min", self.r_min, PLLParamError)
        if not 0.0 <= rmin <= 1.0:
            raise PLLParamError(f"r_min: 需要 0 <= r_min <= 1, 得到 {rmin!r}")

        # 统一规范为 float,避免 int/float 混用造成的表示差异。
        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "ki", ki)
        object.__setattr__(self, "wmax", wmax)
        object.__setattr__(self, "amin", amin)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "jump_eps", jump_eps_norm)
        object.__setattr__(self, "r_min", rmin)


@dataclass(frozen=True)
class JumpCandidate:
    """一次确认的周跳候选事件。

    start_index:    候选起始样本号(第一个 |error| > jump_eps 的有效样本)。
    confirm_index:  确认样本号(第 K 个连续越限有效样本)。
    error_sum:      K 个原始(各自 wrap 到 [-pi, pi))有符号误差的算术累计;
                    跨界时仅供参考, 方向以圆均值为准。
    circular_mean:  候选误差的圆均值(rad);合向量退化时为 None。
    r:              合向量长度 |Σe^{j·e}| / K ∈ [0, 1]。
    ambiguous:      r < r_min 时为 True, 此时 circular_mean=None 且补偿模式
                    不改动 theta。
    errors:         构成候选的 K 个有符号误差(按样本顺序)。
    compensated:    本事件是否实际执行了相位补偿(ambiguous 或仅报告模式为 False)。
    """

    start_index: int
    confirm_index: int
    error_sum: float
    circular_mean: Optional[float]
    r: float
    ambiguous: bool
    errors: Tuple[float, ...]
    compensated: bool


@dataclass(frozen=True)
class Diagnostic:
    """逐样本诊断记录。

    index:     样本序号(PLL 实例生命周期内从 0 递增)。
    valid:     幅度是否 >= amin;无效样本冻结更新。
    amplitude: 样本幅度 |z|。
    error:     旋转后的相位误差(rad, [-pi, pi));无效样本为 None。
    omega:     本步之后的频率字(rad/sample);无效样本为冻结值。
    theta:     本步之后的相位(rad, [-pi, pi));无效样本为冻结值;
               启用补偿且本样本确认事件时, 为原更新后再叠加圆均值的相位。
    streak:    本步之后连续满足 |error| <= eps 的有效样本计数
               (确认周跳事件的样本强制为 0)。
    locked:    本步之后是否处于 locked(streak >= lock_count)。
    saturated: 本步 omega 更新是否触发 ±wmax 限幅。
    candidate: 本样本之后是否存在未确认的候选(本样本计入候选且尚未到 K)。
    jump:      本样本确认的 JumpCandidate;未确认时为 None。
    """

    index: int
    valid: bool
    amplitude: float
    error: Optional[float]
    omega: float
    theta: float
    streak: int
    locked: bool
    saturated: bool
    candidate: bool = False
    jump: Optional[JumpCandidate] = None


@dataclass(frozen=True)
class PLLResult:
    """一次 run() 的结果快照。

    omega:        估计频偏(rad/sample)。
    theta:        当前相位(rad, [-pi, pi))。
    locked:       是否处于 locked。
    streak:       当前连续在门限内的有效样本计数。
    lost_samples: 累计失锁(无效)样本数。
    processed:    累计已处理样本数。
    diagnostics:  本批逐样本诊断(仅本批,不含历史批次)。
    jumps:        本批确认的周跳事件(仅本批,按确认顺序)。
    jump_count:   PLL 实例生命周期内累计确认的事件数。
    """

    params: PLLParams
    omega: float
    theta: float
    locked: bool
    streak: int
    lost_samples: int
    processed: int
    diagnostics: Tuple[Diagnostic, ...]
    jumps: Tuple[JumpCandidate, ...] = ()
    jump_count: int = 0

    def frequency_offset(self, sample_rate: float = 1.0) -> float:
        """把估计频偏换算到 Hz: omega * sample_rate / (2*pi)。"""
        fs = _finite_real("sample_rate", sample_rate, PLLInputError)
        if fs <= 0.0:
            raise PLLInputError(f"sample_rate: 需要 > 0, 得到 {fs!r}")
        return self.omega * fs / _TWO_PI


class PLL:
    """二阶离散 PLL 捕获器(有状态)。

    同一实例可任意分块连续调用 run()/step(),结果与一次性处理完全一致;
    周跳候选状态同样跨调用保留,分块边界不影响触发结果。

    jump_compensate 选择周跳确认后的行为(仅在启用检测时有意义):
    False 为仅报告(环路更新与禁用检测时完全一致),True 为在确认样本的
    原更新之后按圆均值补偿相位。
    """

    def __init__(self, params: PLLParams, theta0: float = 0.0,
                 omega0: float = 0.0, jump_compensate: bool = False):
        if not isinstance(params, PLLParams):
            raise PLLParamError(
                f"params: 需要 PLLParams, 得到 {type(params).__name__}"
            )
        if not isinstance(jump_compensate, bool):
            raise PLLParamError(
                "jump_compensate: 需要 bool, "
                f"得到 {type(jump_compensate).__name__}({jump_compensate!r})"
            )
        theta0 = _finite_real("theta0", theta0, PLLParamError)
        omega0 = _finite_real("omega0", omega0, PLLParamError)
        if abs(omega0) > params.wmax:
            raise PLLParamError(
                f"omega0: |omega0|={abs(omega0)!r} 超出 wmax={params.wmax!r}, "
                "拒绝静默限幅,请显式调整"
            )
        self._params = params
        self._theta = wrap_phase(theta0)
        self._omega = omega0
        self._streak = 0
        self._lost = 0
        self._processed = 0
        self._jump_compensate = bool(jump_compensate)
        self._jump_total = 0
        # 未确认候选状态(跨 run()/step() 保留)
        self._cand_start: Optional[int] = None
        self._cand_sum = 0.0
        self._cand_count = 0
        self._cand_errors: list[float] = []

    # ---- 只读状态 ----
    @property
    def params(self) -> PLLParams:
        return self._params

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def locked(self) -> bool:
        return self._streak >= self._params.lock_count

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def lost_samples(self) -> int:
        return self._lost

    @property
    def processed(self) -> int:
        return self._processed

    @property
    def jump_compensate(self) -> bool:
        return self._jump_compensate

    @property
    def jump_count(self) -> int:
        return self._jump_total

    @property
    def candidate_active(self) -> bool:
        """当前是否存在跨样本未确认的候选。"""
        return self._cand_start is not None

    def _reset_candidate(self) -> None:
        self._cand_start = None
        self._cand_sum = 0.0
        self._cand_count = 0
        self._cand_errors = []

    # ---- 核心更新 ----
    def _step_validated(self, re: float, im: float) -> Diagnostic:
        p = self._params
        amp = math.hypot(re, im)
        idx = self._processed
        if amp < p.amin:
            # 无效样本:冻结 omega/theta,清零捕获计数,计失锁;
            # 同时打断未确认候选(不确认、不补偿、不伪造相位)。
            self._lost += 1
            self._streak = 0
            self._reset_candidate()
            self._processed += 1
            return Diagnostic(idx, False, amp, None, self._omega, self._theta,
                              0, False, False)
        # 1. 旋转: y = z * exp(-j*theta)
        c = math.cos(self._theta)
        s = math.sin(self._theta)
        y_re = re * c + im * s
        y_im = im * c - re * s
        # 2. 误差 ∈ [-pi, pi)
        error = wrap_phase(math.atan2(y_im, y_re))
        # 3. 频率字更新并限幅
        omega_new = self._omega + p.ki * error
        saturated = False
        if omega_new > p.wmax:
            omega_new = p.wmax
            saturated = True
        elif omega_new < -p.wmax:
            omega_new = -p.wmax
            saturated = True
        # 4. 相位更新(使用限幅后的 omega')
        theta_new = wrap_phase(self._theta + omega_new + p.kp * error)

        # 5. 可选周跳候选检测(作用于本样本的原始误差)
        event: Optional[JumpCandidate] = None
        candidate_active = False
        if p.jump_k is not None:
            if abs(error) > p.jump_eps:
                if self._cand_start is None:
                    self._cand_start = idx
                    self._cand_sum = error
                    self._cand_errors = [error]
                    self._cand_count = 1
                else:
                    self._cand_sum += error
                    self._cand_errors.append(error)
                    self._cand_count += 1
                if self._cand_count >= p.jump_k:
                    mean, r = circular_rmean(self._cand_errors, p.r_min)
                    compensated = False
                    if mean is not None and self._jump_compensate:
                        # 补偿发生在该样本原更新之后;同一样本不再重开候选。
                        theta_new = wrap_phase(theta_new + mean)
                        compensated = True
                    event = JumpCandidate(
                        self._cand_start, idx, self._cand_sum, mean, r,
                        mean is None, tuple(self._cand_errors), compensated,
                    )
                    self._jump_total += 1
                    self._reset_candidate()
                else:
                    candidate_active = True
            else:
                # 未越限的有效样本打断候选,不算确认。
                self._reset_candidate()

        # 6. 捕获计数:确认事件的样本强制清零;其余按 |error| <= eps 判定。
        if event is not None:
            self._streak = 0
        elif abs(error) <= p.eps:
            self._streak += 1
        else:
            self._streak = 0
        self._omega = omega_new
        self._theta = theta_new
        self._processed += 1
        return Diagnostic(idx, True, amp, error, omega_new, theta_new,
                          self._streak, self.locked, saturated,
                          candidate_active, event)

    def step(self, z) -> Diagnostic:
        """处理单个样本并返回其诊断。非法样本抛 PLLInputError,状态不变。"""
        re, im = _coerce_sample(z)
        return self._step_validated(re, im)

    def run(self, samples) -> PLLResult:
        """处理一批样本并返回结果快照。

        先整批校验再逐样本更新:任一样本非法时抛 PLLInputError,
        PLL 状态保持调用前原样(不产生部分变更)。
        """
        if isinstance(samples, (str, bytes)):
            raise PLLInputError("samples: 需要复数样本的可迭代序列, 拒绝字符串")
        try:
            items = list(samples)
        except TypeError:
            raise PLLInputError(
                f"samples: 需要可迭代序列, 得到 {type(samples).__name__}"
            ) from None
        if len(items) > MAX_SAMPLES:
            raise PLLInputError(
                f"samples: 单批最多 {MAX_SAMPLES} 个样本, 得到 {len(items)}"
            )
        coerced = [_coerce_sample(z) for z in items]
        diags = [self._step_validated(re, im) for re, im in coerced]
        jumps = tuple(d.jump for d in diags if d.jump is not None)
        return PLLResult(
            self._params, self._omega, self._theta, self.locked,
            self._streak, self._lost, self._processed, tuple(diags),
            jumps, self._jump_total,
        )
