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
    "DualReferencePLL",
    "DualRefDiagnostic",
    "DualRefResult",
    "MODE_TRACKING",
    "MODE_ACQUIRING",
    "MODE_HOLDOVER",
    "PLLError",
    "PLLParamError",
    "PLLInputError",
    "MAX_SAMPLES",
    "wrap_phase",
]

#: 双参考状态:正在跟踪当前参考(输出可信)。
MODE_TRACKING = "tracking"
#: 双参考状态:当前参考尚未确认锁定(输出不可信)。
MODE_ACQUIRING = "acquiring"
#: 双参考状态:切换/失效期间无任何已确认稳定的参考,输出冻结(不可信)。
MODE_HOLDOVER = "holdover"

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
    """

    params: PLLParams
    omega: float
    theta: float
    locked: bool
    streak: int
    lost_samples: int
    processed: int
    diagnostics: Tuple[Diagnostic, ...]

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

    def __init__(self, params: PLLParams, theta0: float = 0.0, omega0: float = 0.0):
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
        self._params = params
        self._theta = wrap_phase(theta0)
        self._omega = omega0
        self._streak = 0
        self._lost = 0
        self._processed = 0

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

    # ---- 核心更新 ----
    def _step_validated(self, re: float, im: float) -> Diagnostic:
        p = self._params
        amp = math.hypot(re, im)
        idx = self._processed
        if amp < p.amin:
            # 无效样本:冻结 omega/theta,清零捕获计数,计失锁。
            self._lost += 1
            self._streak = 0
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
        return Diagnostic(idx, True, amp, error, omega_new, theta_new,
                          self._streak, self.locked, saturated)

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
        return PLLResult(
            self._params, self._omega, self._theta, self.locked,
            self._streak, self._lost, self._processed, tuple(diags),
        )


@dataclass(frozen=True)
class DualRefDiagnostic:
    """双参考交接器的逐样本诊断。

    index:       样本序号(交接器生命周期内从 0 递增)。
    active:      本步之后输出所依据的参考:0 或 1。
    mode:        本步之后的状态: tracking / acquiring / holdover。
    trusted:     输出相位/频偏是否可信(tracking 时为 True)。
    handoff:     本步是否刚完成一次无扰动交接。
    requested:   本步之后待切换目标:None、0 或 1。
    selected:    当前已交接的参考:None(初始未锁定)、0 或 1。
    theta:       本步之后的输出相位(rad, [-pi, pi))。
    omega:       本步之后的输出频偏(rad/sample)。
    lock0/lock1: 两路参考各自的独立锁定状态。
    valid0/valid1: 两路参考本样本是否有效(幅度 >= amin)。
    offset:      本步活动参考的输出对齐偏置(rad, [-pi, pi))。
    holdover_samples: 本步是否计入保持态失效样本。
    """

    index: int
    active: Optional[int]
    mode: str
    trusted: bool
    handoff: bool
    requested: Optional[int]
    selected: Optional[int]
    theta: float
    omega: float
    lock0: bool
    lock1: bool
    valid0: bool
    valid1: bool
    offset: float
    holdover_samples: int


@dataclass(frozen=True)
class DualRefResult:
    """双参考交接器一次 run() 的结果快照。

    theta/omega:     当前输出相位/频偏(holdover 时冻结,trusted=False)。
    selected:        当前已交接参考;None 表示尚未有任何参考确认锁定。
    requested:       待切换目标;None 表示没有挂起的切换请求。
    mode:            tracking / acquiring / holdover。
    trusted:         输出是否可信。
    holdover_samples:累计处于 holdover 的样本数。
    handoff_count:   累计完成的交接次数。
    processed:       累计已处理样本对数量。
    diagnostics:     本批逐样本诊断(仅本批)。
    """

    theta: float
    omega: float
    selected: Optional[int]
    requested: Optional[int]
    mode: str
    trusted: bool
    holdover_samples: int
    handoff_count: int
    processed: int
    diagnostics: Tuple[DualRefDiagnostic, ...]


class DualReferencePLL:
    """同采样速率两路参考之间的无扰动切换交接器。

    两路参考各自挂一个独立的 :class:`PLL`,每个样本对两路同时、独立更新。
    输出相位不是某一路的裸相位,而是 ``wrap(theta_i + offset_i)``:

    - 目标参考先独立跟踪,连续 L 个样本误差在门限内(其 PLL 进入 locked)
      才确认稳定;未稳定前继续输出原参考。
    - 交接时仅依据交接瞬间两路裸相位重新计算**一次性**对齐偏置:
      ``offset_new = wrap(theta_old_out - theta_new_raw)``,使交接样本的
      输出相位保持原值,不发生跳变;偏置每次交接都重新从零推导,
      故来回切换不会反复累积人为相位偏移。
    - 切换期间当前参考失效(幅度 < amin)且目标尚未稳定时进入 holdover:
      冻结输出并明确 ``mode='holdover'``、``trusted=False``,不假装仍锁定。
    """

    def __init__(self, params: PLLParams, theta0: float = 0.0,
                 omega0: float = 0.0):
        # PLL 构造器负责全部参数/初值校验;两路使用同一组环路参数。
        self._trackers = (PLL(params, theta0, omega0),
                          PLL(params, theta0, omega0))
        self._selected = None       # 已交接参考: None/0/1
        self._requested = None      # 待切换目标: None/0/1
        self._offsets = [0.0, 0.0]  # 每路各自的输出对齐偏置
        self._theta = wrap_phase(theta0)
        self._omega = omega0
        self._holdover = 0
        self._handoffs = 0
        self._processed = 0
        self._ever_locked = False   # 是否曾有参考确认锁定(区分 acquiring/holdover)

    # ---- 只读状态 ----
    @property
    def params(self) -> PLLParams:
        return self._trackers[0].params

    @property
    def trackers(self) -> Tuple[PLL, PLL]:
        return self._trackers

    @property
    def selected(self) -> Optional[int]:
        return self._selected

    @property
    def requested(self) -> Optional[int]:
        return self._requested

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def trusted(self) -> bool:
        return self._selected is not None

    @property
    def mode(self) -> str:
        if self._selected is None:
            return MODE_HOLDOVER if self._ever_locked else MODE_ACQUIRING
        return MODE_TRACKING

    @property
    def holdover_samples(self) -> int:
        return self._holdover

    @property
    def handoff_count(self) -> int:
        return self._handoffs

    @property
    def processed(self) -> int:
        return self._processed

    def offset(self, ref: int) -> float:
        """返回指定参考(0/1)当前的输出对齐偏置。"""
        if not isinstance(ref, int) or isinstance(ref, bool) or ref not in (0, 1):
            raise PLLParamError(f"ref: 需要 0 或 1, 得到 {ref!r}")
        return self._offsets[ref]

    def request_switch(self, ref: int) -> None:
        """请求切换到参考 ref(0/1)。

        目标参考独立跟踪并确认稳定(PLL locked)后,在下一个有效样本完成
        无扰动交接;未稳定前继续输出当前参考。非法 ref 抛 PLLParamError。
        """
        if not isinstance(ref, int) or isinstance(ref, bool) or ref not in (0, 1):
            raise PLLParamError(f"ref: 需要 0 或 1, 得到 {ref!r}")
        self._requested = ref

    def cancel_switch(self) -> None:
        """撤销尚未完成的切换请求。"""
        self._requested = None

    def _snapshot(self):
        """保存全部状态,供整批校验失败时回滚(批次原子性)。"""
        t0, t1 = self._trackers
        return (
            (t0._theta, t0._omega, t0._streak, t0._lost, t0._processed),
            (t1._theta, t1._omega, t1._streak, t1._lost, t1._processed),
            self._selected, self._requested,
            (self._offsets[0], self._offsets[1]),
            self._theta, self._omega, self._holdover,
            self._handoffs, self._processed, self._ever_locked,
        )

    def _restore(self, snap) -> None:
        (s0, s1, selected, requested, offsets, theta, omega, holdover,
         handoffs, processed, ever_locked) = snap
        for tracker, state in zip(self._trackers, (s0, s1)):
            tracker._theta, tracker._omega, tracker._streak, tracker._lost, tracker._processed = state
        self._selected = selected
        self._requested = requested
        self._offsets[0], self._offsets[1] = offsets
        self._theta = theta
        self._omega = omega
        self._holdover = holdover
        self._handoffs = handoffs
        self._processed = processed
        self._ever_locked = ever_locked

    def _take_handoff(self, target: int) -> None:
        """在更新后的两路状态上完成一次无扰动交接。

        输出相位保持上一样本值(theta_old),仅按目标参考的新裸相位
        重新计算一次性偏置;不修改、不继承旧偏置,因此反复切换不累积。
        """
        theta_old = self._theta
        theta_new_raw = self._trackers[target].theta
        if self._holdover == 0 and self._selected is None and self._handoffs == 0:
            # 系统首次接管:输出直接对齐该参考裸相位,不引入人为偏置。
            offset = 0.0
            theta_out = theta_new_raw
        else:
            # 后续交接:仅按两路绝对相位一次性重算偏置,不继承旧偏置。
            offset = wrap_phase(theta_old - theta_new_raw)
            theta_out = theta_old
        self._offsets[target] = offset
        self._selected = target
        self._requested = None
        self._theta = theta_out
        self._omega = self._trackers[target].omega
        self._handoffs += 1
        self._ever_locked = True

    def step(self, pair) -> DualRefDiagnostic:
        """处理一个样本对 ``(z0, z1)`` 并返回诊断。

        两路参考同采样速率,每步成对喂入。非法样本抛 PLLInputError,
        交接器与两路 PLL 状态均保持调用前原样。
        """
        if isinstance(pair, (str, bytes)) or not isinstance(pair, tuple):
            raise PLLInputError(
                "pair: 需要长度为 2 的样本对 (z0, z1), "
                f"得到 {type(pair).__name__}"
            )
        if len(pair) != 2:
            raise PLLInputError(f"pair: 需要长度为 2, 得到长度 {len(pair)}")
        coerced0 = _coerce_sample(pair[0])
        coerced1 = _coerce_sample(pair[1])
        return self._step_validated(coerced0, coerced1)

    def _step_validated(self, z0, z1) -> DualRefDiagnostic:
        t0, t1 = self._trackers
        d0 = t0._step_validated(*z0)
        d1 = t1._step_validated(*z1)
        locked = (d0.locked, d1.locked)
        valid = (d0.valid, d1.valid)

        handoff = False

        # 1) 挂起的切换目标已确认稳定:完成无扰动交接。
        target = self._requested
        if target is not None and locked[target] and valid[target]:
            self._take_handoff(target)
            handoff = True

        active = self._selected
        in_holdover_step = False

        if not handoff and active is not None and (d0, d1)[active].valid:
            # 2) 活动参考存在且本样本有效:输出随活动参考连续推进(偏置固定)。
            tracker = (t0, t1)[active]
            self._theta = wrap_phase(tracker.theta + self._offsets[active])
            self._omega = tracker.omega
        elif not handoff:
            # 3) 活动参考缺失或本样本失效:优先接管已确认稳定且有效的
            #    参考(挂起目标优先,其次任一路);没有则保持/捕获。
            candidate = None
            if target is not None and locked[target] and valid[target]:
                candidate = target
            elif active is None:
                for ref in (0, 1):
                    if locked[ref] and valid[ref]:
                        candidate = ref
                        break
            # 注:active 非 None 但失效、且无挂起目标时不自动切到另一路,
            # 必须等待活动参考重新确认(显式切换语义),此期间进入保持态。
            if candidate is not None:
                self._take_handoff(candidate)
                handoff = True
                active = self._selected
            else:
                if active is not None:
                    self._selected = None  # 活动参考本样本失效
                    active = None
                if self._ever_locked:
                    self._holdover += 1
                    in_holdover_step = True

        if active is not None:
            mode = MODE_TRACKING
        elif self._ever_locked:
            # 曾有可信参考,当前参考失效且无稳定目标:保持态(不可信)。
            mode = MODE_HOLDOVER
        else:
            # 自启动以来尚无任何参考确认锁定:捕获态(同样不可信)。
            mode = MODE_ACQUIRING

        self._processed += 1
        return DualRefDiagnostic(
            self._processed - 1, active, mode, active is not None,
            handoff, self._requested, self._selected,
            self._theta, self._omega,
            d0.locked, d1.locked, d0.valid, d1.valid,
            self._offsets[active] if active is not None else 0.0,
            1 if in_holdover_step else 0,
        )

    def run(self, pairs) -> DualRefResult:
        """处理一批样本对,返回结果快照。

        先整批校验再逐对更新:任一样本非法时抛 PLLInputError,
        交接器与两路 PLL 状态保持调用前原样(批次原子性)。
        """
        if isinstance(pairs, (str, bytes)):
            raise PLLInputError("pairs: 需要样本对序列, 拒绝字符串")
        try:
            items = list(pairs)
        except TypeError:
            raise PLLInputError(
                f"pairs: 需要可迭代序列, 得到 {type(pairs).__name__}"
            ) from None
        if len(items) > MAX_SAMPLES:
            raise PLLInputError(
                f"pairs: 单批最多 {MAX_SAMPLES} 个样本对, 得到 {len(items)}"
            )
        coerced = []
        snap = self._snapshot()
        try:
            for i, pair in enumerate(items):
                if isinstance(pair, (str, bytes)) or not isinstance(pair, tuple):
                    raise PLLInputError(
                        f"pairs[{i}]: 需要长度为 2 的样本对, "
                        f"得到 {type(pair).__name__}"
                    )
                if len(pair) != 2:
                    raise PLLInputError(
                        f"pairs[{i}]: 需要长度为 2, 得到长度 {len(pair)}"
                    )
                coerced.append((_coerce_sample(pair[0]), _coerce_sample(pair[1])))
            diags = [self._step_validated(z0, z1) for z0, z1 in coerced]
        except BaseException:
            self._restore(snap)
            raise
        return DualRefResult(
            self._theta, self._omega, self._selected, self._requested,
            self.mode, self.trusted, self._holdover, self._handoffs,
            self._processed, tuple(diags),
        )
