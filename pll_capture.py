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

可选的保持(holdover)/重捕获(reacquire)状态机(默认关闭, 关闭时行为与
上述基础契约逐位一致):

- tracking:  正常跟踪(基础契约行为)。
- holdover:  幅度连续 < amin 达到 holdover_enter 个样本后进入。
             冻结 omega 为最近 holdover_window 个有效频率估计的中位数,
             每个样本仅以该冻结频率推进相位, 不用低幅噪声更新积分项。
- reacquire: 参考重新有效后进入。每样本相位校正项被限制在
             ±reacquire_step 内逐渐追上, 不瞬间重置相位; 连续
             reacquire_count 个样本同时满足 |error| <= eps 且
             |omega - omega_hold| <= freq_eps 才回到 tracking 并重新
             标记锁定。重捕获期间再次出现低幅样本立即回到 holdover。

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
    "MAX_SAMPLES",
    "STATE_TRACKING",
    "STATE_HOLDOVER",
    "STATE_REACQUIRE",
    "wrap_phase",
]

#: run() 单次调用允许的最大样本数(诊断逐样本保存,限制内存占用)。
MAX_SAMPLES = 1_000_000

#: PLL 状态机三个状态的名称(字符串常量, 出现在 Diagnostic.state 中)。
STATE_TRACKING = "tracking"
STATE_HOLDOVER = "holdover"
STATE_REACQUIRE = "reacquire"

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


def _median(values) -> float:
    """有限实数序列的中位数; 偶数个取中间两值的均值(确定性)。"""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


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

    可选保持/重捕获参数(默认值保持基础契约行为, 即低幅样本完全冻结):

    holdover_enter: 连续低幅样本达到该数后进入 holdover;None 表示禁用
                    (默认, 兼容旧行为)。int 且 >= 1(拒绝 bool)。
    holdover_window:holdover 冻结频率取最近该数个有效 omega 的中位数,
                    int 且 >= 1。
    reacquire_step: reacquire 期间每样本相位校正项 |kp*error| 的上限(rad),
                    有限且 > 0。
    freq_eps:       重新锁定的频率误差容限 |omega - omega_hold|(rad/sample),
                    有限且 > 0。
    reacquire_count:重新锁定所需的连续满足双条件样本数,int 且 >= 1。
    """

    kp: float
    ki: float
    wmax: float
    amin: float
    eps: float
    lock_count: int
    holdover_enter: Optional[int] = None
    holdover_window: int = 16
    reacquire_step: float = 0.05
    freq_eps: float = 0.01
    reacquire_count: int = 20

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
        he = self.holdover_enter
        if he is not None:
            if isinstance(he, bool) or not isinstance(he, int):
                raise PLLParamError(
                    f"holdover_enter: 需要 int 或 None(拒绝 bool), "
                    f"得到 {type(he).__name__}({he!r})"
                )
            if he < 1:
                raise PLLParamError(f"holdover_enter: 需要 >= 1, 得到 {he!r}")
        hw = self.holdover_window
        if isinstance(hw, bool) or not isinstance(hw, int):
            raise PLLParamError(
                f"holdover_window: 需要 int(拒绝 bool), "
                f"得到 {type(hw).__name__}({hw!r})"
            )
        if hw < 1:
            raise PLLParamError(f"holdover_window: 需要 >= 1, 得到 {hw!r}")
        rc = self.reacquire_count
        if isinstance(rc, bool) or not isinstance(rc, int):
            raise PLLParamError(
                f"reacquire_count: 需要 int(拒绝 bool), "
                f"得到 {type(rc).__name__}({rc!r})"
            )
        if rc < 1:
            raise PLLParamError(f"reacquire_count: 需要 >= 1, 得到 {rc!r}")
        reacquire_step = _finite_real("reacquire_step", self.reacquire_step,
                                      PLLParamError)
        freq_eps = _finite_real("freq_eps", self.freq_eps, PLLParamError)
        if not reacquire_step > 0.0:
            raise PLLParamError(
                f"reacquire_step: 需要 > 0, 得到 {reacquire_step!r}")
        if not freq_eps > 0.0:
            raise PLLParamError(f"freq_eps: 需要 > 0, 得到 {freq_eps!r}")
        # 统一规范为 float,避免 int/float 混用造成的表示差异。
        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "ki", ki)
        object.__setattr__(self, "wmax", wmax)
        object.__setattr__(self, "amin", amin)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "reacquire_step", reacquire_step)
        object.__setattr__(self, "freq_eps", freq_eps)


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
    state:     本步之后的状态: "tracking" / "holdover" / "reacquire"。
               (holdover 禁用且样本有效时恒为 "tracking";
               holdover 禁用时的低幅冻结样本也记为 "tracking"。)
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
    state: str = STATE_TRACKING


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
    state:        本批处理后的状态机状态。
    """

    params: PLLParams
    omega: float
    theta: float
    locked: bool
    streak: int
    lost_samples: int
    processed: int
    diagnostics: Tuple[Diagnostic, ...]
    state: str = STATE_TRACKING

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
        # 保持/重捕获状态机(holdover_enter=None 时恒为 tracking)。
        self._state = STATE_TRACKING
        self._low_amp = 0            # 连续低幅样本计数(跨 chunk 延续)
        self._omega_hold = 0.0       # holdover 冻结频率(中位数)
        self._omega_hist = []        # 最近 holdover_window 个有效 omega
        self._reacquire_streak = 0   # 连续满足重捕获双条件的样本数

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
        # 只有 tracking 状态下连续计数达标才算锁定;
        # holdover/reacquire 期间即使 streak 达标也不标记 locked。
        return (self._state == STATE_TRACKING
                and self._streak >= self._params.lock_count)

    @property
    def state(self) -> str:
        return self._state

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
            self._lost += 1
            self._streak = 0
            self._low_amp += 1
            if p.holdover_enter is not None and (
                    self._state == STATE_HOLDOVER
                    or self._state == STATE_REACQUIRE
                    or self._low_amp >= p.holdover_enter):
                # 进入/保持 holdover: 冻结频率为最近有效估计的中位数,
                # 仅以冻结频率推进相位, 不用低幅噪声更新积分项。
                # 注: reacquire 期间出现低幅样本立即回到 holdover。
                if self._state != STATE_HOLDOVER:
                    self._omega_hold = (_median(self._omega_hist)
                                        if self._omega_hist else self._omega)
                    self._state = STATE_HOLDOVER
                self._omega = self._omega_hold
                self._theta = wrap_phase(self._theta + self._omega_hold)
                self._processed += 1
                return Diagnostic(idx, False, amp, None, self._omega,
                                  self._theta, 0, False, False,
                                  STATE_HOLDOVER)
            # 无效样本(holdover 未启用或未达门限):冻结 omega/theta,
            # 清零捕获计数,计失锁。
            self._processed += 1
            return Diagnostic(idx, False, amp, None, self._omega, self._theta,
                              0, False, False, self._state)
        self._low_amp = 0
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
        if self._state == STATE_HOLDOVER:
            # 参考重新有效: 进入 reacquire, 不瞬间重置相位。
            self._state = STATE_REACQUIRE
            self._reacquire_streak = 0
        if self._state == STATE_REACQUIRE:
            # 4'. 有界校正: 相位 = 频率前馈 + 限幅后的校正项,
            # 每样本校正量 |kp*error| 不超过 reacquire_step, 逐渐追上。
            correction = p.kp * error
            if correction > p.reacquire_step:
                correction = p.reacquire_step
            elif correction < -p.reacquire_step:
                correction = -p.reacquire_step
            theta_new = wrap_phase(self._theta + omega_new + correction)
            # 双条件: 相位误差与频率误差同时达标才累计; 任一超限清零。
            if (abs(error) <= p.eps
                    and abs(omega_new - self._omega_hold) <= p.freq_eps):
                self._reacquire_streak += 1
            else:
                self._reacquire_streak = 0
            if self._reacquire_streak >= p.reacquire_count:
                self._state = STATE_TRACKING
        else:
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
        # 记录最近 holdover_window 个有效频率估计(供 holdover 取中位数)
        self._omega_hist.append(omega_new)
        if len(self._omega_hist) > p.holdover_window:
            del self._omega_hist[0]
        return Diagnostic(idx, True, amp, error, omega_new, theta_new,
                          self._streak, self.locked, saturated, self._state)

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
            self._state,
        )
