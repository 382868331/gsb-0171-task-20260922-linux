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

可选的周跳候选检测(传入 CycleSlipConfig 启用):

- 有效样本 |error| 连续 confirm_count 次超过 threshold 时确认一次事件;
  事件保留起始样本号、确认号与累计有符号误差。
- 低幅度(无效)样本打断候选:不累计、不确认,也不伪造有效相位(error 仍为 None)。
- 确认后锁定计数清零、频率估计保留;按配置仅报告,或在该样本原更新之后
  把相位补偿候选误差的圆均值。合向量长度(平均 resultant)低于 deg_threshold
  时判定 ambiguous:只报告,不挑选补偿方向。
- 确认候选的那个样本不会再次开启新候选,防止连发。
- 候选状态保存在实例内,任意分块处理结果一致。

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
    "PLLError",
    "PLLParamError",
    "PLLInputError",
    "CycleSlipConfig",
    "CycleSlipEvent",
    "MAX_SAMPLES",
    "wrap_phase",
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
    """

    kp: float
    ki: float
    wmax: float
    amin: float
    eps: float
    lock_count: int

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
        # 统一规范为 float,避免 int/float 混用造成的表示差异。
        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "ki", ki)
        object.__setattr__(self, "wmax", wmax)
        object.__setattr__(self, "amin", amin)
        object.__setattr__(self, "eps", eps)


@dataclass(frozen=True)
class CycleSlipConfig:
    """周跳候选检测配置(可选;PLL 不传 slip_config 即完全关闭)。

    threshold:     候选误差门限(rad),有限且 0 < threshold <= pi。
                   有效样本 |error| 超过它才计入候选。
    confirm_count: 确认所需的连续超限次数 K,int 且 >= 1(拒绝 bool)。
    deg_threshold: 圆均值退化阈值,有限且 0 <= deg_threshold <= 1。
                   候选误差的平均合向量长度低于它时判定 ambiguous,
                   只报告、不补偿(不随便挑方向)。
    compensate:    True 时确认后在原更新之后把相位补偿候选误差的圆均值;
                   False 时仅报告。必须显式给 bool。
    """

    threshold: float
    confirm_count: int
    deg_threshold: float
    compensate: bool

    def __post_init__(self) -> None:
        threshold = _finite_real("threshold", self.threshold, PLLParamError)
        if not 0.0 < threshold <= math.pi:
            raise PLLParamError(
                f"threshold: 需要 0 < threshold <= pi, 得到 {threshold!r}"
            )
        k = self.confirm_count
        if isinstance(k, bool) or not isinstance(k, int):
            raise PLLParamError(
                f"confirm_count: 需要 int(拒绝 bool), 得到 {type(k).__name__}({k!r})"
            )
        if k < 1:
            raise PLLParamError(f"confirm_count: 需要 >= 1, 得到 {k!r}")
        deg = _finite_real("deg_threshold", self.deg_threshold, PLLParamError)
        if not 0.0 <= deg <= 1.0:
            raise PLLParamError(
                f"deg_threshold: 需要 0 <= deg_threshold <= 1, 得到 {deg!r}"
            )
        if not isinstance(self.compensate, bool):
            raise PLLParamError(
                f"compensate: 需要显式 bool, 得到 "
                f"{type(self.compensate).__name__}({self.compensate!r})"
            )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "deg_threshold", deg)


@dataclass(frozen=True)
class CycleSlipEvent:
    """一次确认的周跳事件。

    start_index:   候选起始样本号(首次超限的有效样本)。
    confirm_index: 确认号(第 K 次连续超限的有效样本)。
    count:         计入本事件的连续超限样本数(== confirm_count)。
    error_sum:     候选误差的累计有符号和(rad,按 [-pi, pi) 原样累加,
                   跨 ±pi 时不代表真实偏移,仅供记录)。
    circular_mean: 候选误差的圆均值(rad);ambiguous 时为 None(不挑方向)。
    resultant:     候选误差单位向量的平均合向量长度,取值 [0, 1]。
    ambiguous:     resultant < deg_threshold,方向不可信,未补偿。
    compensated:   是否已把相位补偿 circular_mean。
    correction:    实际施加的相位补偿量(rad);未补偿为 0.0。
    """

    start_index: int
    confirm_index: int
    count: int
    error_sum: float
    circular_mean: Optional[float]
    resultant: float
    ambiguous: bool
    compensated: bool
    correction: float


@dataclass(frozen=True)
class Diagnostic:
    """逐样本诊断记录。

    index:     样本序号(PLL 实例生命周期内从 0 递增)。
    valid:     幅度是否 >= amin;无效样本冻结更新。
    amplitude: 样本幅度 |z|。
    error:     旋转后的相位误差(rad, [-pi, pi));无效样本为 None。
    omega:     本步之后的频率字(rad/sample);无效样本为冻结值。
    theta:     本步之后的相位(rad, [-pi, pi));无效样本为冻结值。
    streak:    本步之后连续满足 |error| <= eps 的有效样本计数。
    locked:    本步之后是否处于 locked(streak >= lock_count)。
    saturated: 本步 omega 更新是否触发 ±wmax 限幅。
    slip_event: 本样本确认的周跳事件;未确认或未启用检测为 None。
                确认时 theta/streak 记录的是补偿与清零之后的值。
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
    slip_event: Optional[CycleSlipEvent] = None


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
    slip_events:  本批确认的周跳事件(未启用检测或本批无确认为空元组)。
    """

    params: PLLParams
    omega: float
    theta: float
    locked: bool
    streak: int
    lost_samples: int
    processed: int
    diagnostics: Tuple[Diagnostic, ...]
    slip_events: Tuple[CycleSlipEvent, ...] = ()

    def frequency_offset(self, sample_rate: float = 1.0) -> float:
        """把估计频偏换算到 Hz: omega * sample_rate / (2*pi)。"""
        fs = _finite_real("sample_rate", sample_rate, PLLInputError)
        if fs <= 0.0:
            raise PLLInputError(f"sample_rate: 需要 > 0, 得到 {fs!r}")
        return self.omega * fs / _TWO_PI


class PLL:
    """二阶离散 PLL 捕获器(有状态)。

    同一实例可任意分块连续调用 run()/step(),结果与一次性处理完全一致。
    """

    def __init__(self, params: PLLParams, theta0: float = 0.0, omega0: float = 0.0,
                 slip_config: Optional[CycleSlipConfig] = None):
        if not isinstance(params, PLLParams):
            raise PLLParamError(
                f"params: 需要 PLLParams, 得到 {type(params).__name__}"
            )
        theta0 = _finite_real("theta0", theta0, PLLParamError)
        omega0 = _finite_real("omega0", omega0, PLLParamError)
        if abs(omega0) > params.wmax:
            raise PLLParamError(
                f"omega0: |omega0|={abs(omega0)!r} 超出 wmax={params.wmax!r}, "
                "拒绝静默限幅,请显式调整"
            )
        if slip_config is not None and not isinstance(slip_config, CycleSlipConfig):
            raise PLLParamError(
                f"slip_config: 需要 CycleSlipConfig 或 None, 得到 "
                f"{type(slip_config).__name__}"
            )
        self._params = params
        self._theta = wrap_phase(theta0)
        self._omega = omega0
        self._streak = 0
        self._lost = 0
        self._processed = 0
        # 周跳候选检测状态(随实例跨分块保存,保证任意切块结果一致)。
        self._slip = slip_config
        self._cand_start: Optional[int] = None
        self._cand_count = 0
        self._cand_err_sum = 0.0
        self._cand_sin = 0.0
        self._cand_cos = 0.0
        self._slip_events = []

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
    def slip_events(self) -> Tuple[CycleSlipEvent, ...]:
        """实例生命周期内累计确认的周跳事件。"""
        return tuple(self._slip_events)

    # ---- 周跳候选检测 ----
    def _reset_candidate(self) -> None:
        self._cand_start = None
        self._cand_count = 0
        self._cand_err_sum = 0.0
        self._cand_sin = 0.0
        self._cand_cos = 0.0

    def _update_slip(self, idx: int, error: float) -> Optional[CycleSlipEvent]:
        """在一个有效样本的原更新之后推进候选状态;确认时返回事件。

        确认候选的那个样本已把误差计入本事件并清零候选,不会再次开启
        新候选(防止连发);下一个候选最早从下一个样本开始。
        """
        cfg = self._slip
        if abs(error) > cfg.threshold:
            if self._cand_count == 0:
                self._cand_start = idx
            self._cand_count += 1
            self._cand_err_sum += error
            self._cand_sin += math.sin(error)
            self._cand_cos += math.cos(error)
            if self._cand_count >= cfg.confirm_count:
                event = self._confirm_slip(idx)
                self._reset_candidate()
                return event
        else:
            # 有效样本误差回到门限内:打断候选。
            self._reset_candidate()
        return None

    def _confirm_slip(self, idx: int) -> CycleSlipEvent:
        """确认一次事件:锁定计数清零、频率估计保留,按配置补偿相位。"""
        cfg = self._slip
        k = self._cand_count
        mean_sin = self._cand_sin / k
        mean_cos = self._cand_cos / k
        resultant = math.hypot(mean_sin, mean_cos)
        ambiguous = resultant < cfg.deg_threshold
        # 合向量接近零时方向不可信:不计算补偿方向,只报告 ambiguous。
        circular_mean = None if ambiguous else math.atan2(mean_sin, mean_cos)
        correction = 0.0
        compensated = False
        # 锁定计数清零;omega(频率估计)保持原更新后的值,不做任何改动。
        self._streak = 0
        if cfg.compensate and not ambiguous:
            correction = circular_mean
            self._theta = wrap_phase(self._theta + correction)
            compensated = True
        event = CycleSlipEvent(
            start_index=self._cand_start,
            confirm_index=idx,
            count=k,
            error_sum=self._cand_err_sum,
            circular_mean=circular_mean,
            resultant=resultant,
            ambiguous=ambiguous,
            compensated=compensated,
            correction=correction,
        )
        self._slip_events.append(event)
        return event

    # ---- 核心更新 ----
    def _step_validated(self, re: float, im: float) -> Diagnostic:
        p = self._params
        amp = math.hypot(re, im)
        idx = self._processed
        if amp < p.amin:
            # 无效样本:冻结 omega/theta,清零捕获计数,计失锁;
            # 同时打断周跳候选(不累计、不确认,也不伪造有效相位)。
            self._lost += 1
            self._streak = 0
            if self._slip is not None:
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
        # 捕获计数:超限立即清零
        if abs(error) <= p.eps:
            self._streak += 1
        else:
            self._streak = 0
        self._omega = omega_new
        self._theta = theta_new
        self._processed += 1
        # 周跳候选检测在原更新之后推进;确认时的补偿也发生在原更新之后。
        slip_event = None
        if self._slip is not None:
            slip_event = self._update_slip(idx, error)
        return Diagnostic(idx, True, amp, error, self._omega, self._theta,
                          self._streak, self.locked, saturated, slip_event)

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
        batch_events = tuple(d.slip_event for d in diags
                             if d.slip_event is not None)
        return PLLResult(
            self._params, self._omega, self._theta, self.locked,
            self._streak, self._lost, self._processed, tuple(diags),
            batch_events,
        )
