"""整数采样数字锁相环(PLL)捕获诊断模块(含 holdover/reacquire 状态机)。

基于合成复数导频的二阶离散 PLL。每个有效样本按如下顺序更新:

1. 旋转:   y = z * exp(-j*theta)
2. 误差:   error = wrap(atan2(Im y, Re y))  ∈ [-pi, pi)
3. 频率:   omega' = clip(omega + ki*error, -wmax, +wmax)
4. 相位:   theta' = wrap(theta + omega' + kp*error)

幅度 < amin 的样本视为无效:冻结 omega/theta 更新、清零捕获计数、
计一次失锁并在诊断中标记 valid=False。

连续 lock_count 个有效样本满足 |error| <= eps 时进入 locked;
任一有效样本 |error| > eps 立即清零计数并退出 locked。

可选的导频短暂消失处理(默认关闭, 行为与旧版完全一致):

- tracking:  正常跟踪(默认状态)。
- holdover:  幅度连续低于 amin 达到 holdover_samples 个样本后进入;
             冻结最近 median_window 个有效频率估计的中位数 omega_hold,
             每样本仅以 theta += omega_hold 推进相位, 不再用噪声更新积分项。
- reacquire: 参考重新有效后进入; 不瞬间重置相位, 每样本相位校正被限幅在
             ±max_phase_step 内逐渐追上; 连续 reacquire_count 个样本同时满足
             |error| <= eps 且 |omega' - omega| <= freq_eps 才重新标记锁定
             并回到 tracking。

门限相等时的归属(明确约定): 幅度 == amin 判为有效样本; |error| == eps 计入
捕获/重捕获计数; |omega' - omega| == freq_eps 计入重捕获计数(均为 <= 语义)。

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
    "wrap_phase",
]

#: 环路状态字面量
STATE_TRACKING = "tracking"
STATE_HOLDOVER = "holdover"
STATE_REACQUIRE = "reacquire"

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
    holdover_samples: 连续低幅样本达到该数后进入 holdover;0 表示禁用(默认,
                保持旧版"冻结并清零计数"行为)。int 且 >= 0(拒绝 bool)。
    median_window:  holdover 冻结频率取最近 W 个有效样本 omega 的中位数,W >= 1。
    max_phase_step: reacquire 期间每样本相位校正幅度上限(rad),有限且 > 0。
    freq_eps:   reacquire 重锁定的频率误差条件 |omega' - omega| <= freq_eps,> 0。
    reacquire_count: 重锁定所需的连续满足双条件的样本数,int 且 >= 1。
    """

    kp: float
    ki: float
    wmax: float
    amin: float
    eps: float
    lock_count: int
    holdover_samples: int = 0
    median_window: int = 5
    max_phase_step: float = 0.05
    freq_eps: float = 0.01
    reacquire_count: int = 10

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
        for name in ("holdover_samples", "median_window", "reacquire_count"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise PLLParamError(
                    f"{name}: 需要 int(拒绝 bool), 得到 {type(v).__name__}({v!r})"
                )
        if self.holdover_samples < 0:
            raise PLLParamError(
                f"holdover_samples: 需要 >= 0, 得到 {self.holdover_samples!r}")
        if self.median_window < 1:
            raise PLLParamError(
                f"median_window: 需要 >= 1, 得到 {self.median_window!r}")
        if self.reacquire_count < 1:
            raise PLLParamError(
                f"reacquire_count: 需要 >= 1, 得到 {self.reacquire_count!r}")
        mps = _finite_real("max_phase_step", self.max_phase_step, PLLParamError)
        feps = _finite_real("freq_eps", self.freq_eps, PLLParamError)
        if not mps > 0.0:
            raise PLLParamError(f"max_phase_step: 需要 > 0, 得到 {mps!r}")
        if not feps > 0.0:
            raise PLLParamError(f"freq_eps: 需要 > 0, 得到 {feps!r}")
        # 统一规范为 float,避免 int/float 混用造成的表示差异。
        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "ki", ki)
        object.__setattr__(self, "wmax", wmax)
        object.__setattr__(self, "amin", amin)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "max_phase_step", mps)
        object.__setattr__(self, "freq_eps", feps)


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
    state:        本批结束时的环路状态。
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
        # holdover/reacquire 状态机(默认参数下恒为 tracking,行为与旧版一致)
        self._state = STATE_TRACKING
        self._low_streak = 0            # 连续低幅样本计数(跨 chunk 延续)
        self._reacquire_streak = 0      # 连续满足重锁定双条件的计数(跨 chunk 延续)
        self._omega_window = []         # 最近 median_window 个有效样本的 omega
        self._omega_hold = 0.0          # holdover 期间冻结的频率字

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
    def state(self) -> str:
        return self._state

    @property
    def omega_hold(self) -> float:
        """holdover 期间冻结的频率字(非 holdover 状态时为最近一次的冻结值)。"""
        return self._omega_hold

    def _enter_holdover(self) -> None:
        window = sorted(self._omega_window)
        n = len(window)
        if n == 0:
            median = self._omega
        elif n % 2 == 1:
            median = window[n // 2]
        else:
            median = (window[n // 2 - 1] + window[n // 2]) / 2.0
        self._omega_hold = median
        self._omega = median
        self._state = STATE_HOLDOVER

    # ---- 核心更新 ----
    def _step_validated(self, re: float, im: float) -> Diagnostic:
        p = self._params
        amp = math.hypot(re, im)
        idx = self._processed
        if amp < p.amin:
            # 无效样本:清零捕获计数,计失锁。
            self._lost += 1
            self._streak = 0
            self._reacquire_streak = 0
            self._processed += 1
            if p.holdover_samples > 0:
                self._low_streak += 1
                if self._state is not STATE_HOLDOVER \
                        and self._low_streak >= p.holdover_samples:
                    self._enter_holdover()
                if self._state is STATE_HOLDOVER:
                    # holdover: 冻结的中位频率继续推进相位, 不用噪声更新积分项。
                    self._theta = wrap_phase(self._theta + self._omega_hold)
                    return Diagnostic(idx, False, amp, None, self._omega,
                                      self._theta, 0, False, False,
                                      STATE_HOLDOVER)
            # 旧版行为(holdover 未启用或未达门限): 冻结 omega/theta。
            return Diagnostic(idx, False, amp, None, self._omega, self._theta,
                              0, False, False, self._state)
        # 有效样本
        self._low_streak = 0
        if self._state is STATE_HOLDOVER:
            # 参考重新有效: 进入 reacquire, 不瞬间重置相位。
            self._state = STATE_REACQUIRE
            self._reacquire_streak = 0
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
        # 4. 相位更新(使用限幅后的 omega'); reacquire 期间校正有界
        if self._state is STATE_REACQUIRE:
            correction = p.kp * error
            if correction > p.max_phase_step:
                correction = p.max_phase_step
            elif correction < -p.max_phase_step:
                correction = -p.max_phase_step
            theta_new = wrap_phase(self._theta + omega_new + correction)
            # 重锁定双条件: 相位误差与频率误差同时连续达标(== 计入, <= 语义)
            if abs(error) <= p.eps \
                    and abs(omega_new - self._omega) <= p.freq_eps:
                self._reacquire_streak += 1
            else:
                self._reacquire_streak = 0
            if self._reacquire_streak >= p.reacquire_count:
                self._state = STATE_TRACKING
                self._streak = p.lock_count  # 重新标记锁定
        else:
            theta_new = wrap_phase(self._theta + omega_new + p.kp * error)
            # 捕获计数:超限立即清零
            if abs(error) <= p.eps:
                self._streak += 1
            else:
                self._streak = 0
        # 记录最近 W 个有效样本的频率字(供 holdover 取中位数)
        self._omega_window.append(omega_new)
        if len(self._omega_window) > p.median_window:
            del self._omega_window[:len(self._omega_window) - p.median_window]
        self._omega = omega_new
        self._theta = theta_new
        self._processed += 1
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
