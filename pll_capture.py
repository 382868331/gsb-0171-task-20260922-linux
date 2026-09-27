"""整数采样数字锁相环(PLL)捕获诊断与双参考切换模块。

基于合成复数导频的二阶离散 PLL。每个有效样本按如下顺序更新:

1. 旋转:   y = z * exp(-j*theta)
2. 误差:   error = wrap(atan2(Im y, Re y))  ∈ [-pi, pi)
3. 频率:   omega' = clip(omega + ki*error, -wmax, +wmax)
4. 相位:   theta' = wrap(theta + omega' + kp*error)

幅度 < amin 的样本视为无效:冻结 omega/theta 更新、清零捕获计数、
计一次失锁并在诊断中标记 valid=False。

连续 lock_count 个有效样本满足 |error| <= eps 时进入 locked;
任一有效样本 |error| > eps 立即清零计数并退出 locked。

ReferenceSwitcher 在相同采样速率的两路参考之间切换:
新参考由独立的候选环路先行捕获,确认 locked 后才交接;
交接瞬间输出相位取在用环路的当前相位(不跳变),
频率字取候选环路的收敛估计,不叠加任何人为相位偏移,
因此来回切换不会累积相位误差。目标参考未稳定时 complete_switch()
拒绝交接并继续跟踪原参考;切换期间原参考失效时 locked 如实为 False。

仅依赖 Python 标准库(math / dataclasses)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "PLLParams",
    "PLL",
    "ReferenceSwitcher",
    "Diagnostic",
    "PLLResult",
    "PLLError",
    "PLLParamError",
    "PLLInputError",
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


class ReferenceSwitcher:
    """相同采样速率的两路参考之间的相位连续切换器。

    - 在用环路跟踪当前参考;begin_switch() 后候选环路以在用环路的
      当前 theta/omega 为初值,独立捕获新参考(两路采样速率相同,
      频率字单位同为 rad/sample,可直接复用)。
    - complete_switch() 仅当候选环路 locked 才交接:新在用环路
      theta 取交接前在用环路的当前相位(输出相位不跳变),
      omega 取候选环路的收敛估计。不注入任何人为相位偏移,
      反复来回切换不会累积相位误差。
    - 候选未稳定时 complete_switch() 返回 False,继续跟踪原参考。
    - 切换期间原参考出现无效样本时,locked 如实变为 False,
      status 报告 "switching",不假装仍锁定。
    """

    STATUS_LOCKED = "locked"
    STATUS_ACQUIRING = "acquiring"
    STATUS_SWITCHING = "switching"

    def __init__(self, params: PLLParams, theta0: float = 0.0, omega0: float = 0.0):
        self._active = PLL(params, theta0=theta0, omega0=omega0)
        self._candidate: Optional[PLL] = None
        self._switch_count = 0

    # ---- 只读状态 ----
    @property
    def params(self) -> PLLParams:
        return self._active.params

    @property
    def theta(self) -> float:
        """输出相位(在用环路当前相位, rad, [-pi, pi))。"""
        return self._active.theta

    @property
    def omega(self) -> float:
        return self._active.omega

    @property
    def locked(self) -> bool:
        """在用环路是否 locked;切换期间原参考失效时如实为 False。"""
        return self._active.locked

    @property
    def switching(self) -> bool:
        """是否处于切换过程(候选环路已启动、尚未交接或中止)。"""
        return self._candidate is not None

    @property
    def candidate_locked(self) -> bool:
        """候选环路是否已稳定锁定新参考。"""
        return self._candidate is not None and self._candidate.locked

    @property
    def switch_count(self) -> int:
        """已完成的交接次数。"""
        return self._switch_count

    @property
    def status(self) -> str:
        """可信状态: "locked" / "acquiring" / "switching"。"""
        if self._candidate is not None:
            return self.STATUS_SWITCHING
        return self.STATUS_LOCKED if self._active.locked else self.STATUS_ACQUIRING

    # ---- 在用参考(原参考)输入 ----
    def step(self, z) -> Diagnostic:
        """处理在用参考的单个样本。"""
        return self._active.step(z)

    def run(self, samples) -> PLLResult:
        """处理在用参考的一批样本(批次原子性同 PLL.run)。"""
        return self._active.run(samples)

    # ---- 切换流程 ----
    def begin_switch(self) -> None:
        """启动切换:候选环路以在用环路当前 theta/omega 为初值,
        开始独立跟踪新参考。重复调用会重新初始化候选环路。"""
        self._candidate = PLL(
            self._active.params,
            theta0=self._active.theta,
            omega0=self._active.omega,
        )

    def step_candidate(self, z) -> Diagnostic:
        """向候选环路喂入新参考的单个样本。"""
        if self._candidate is None:
            raise PLLError("step_candidate: 尚未调用 begin_switch()")
        return self._candidate.step(z)

    def run_candidate(self, samples) -> PLLResult:
        """向候选环路喂入新参考的一批样本。"""
        if self._candidate is None:
            raise PLLError("run_candidate: 尚未调用 begin_switch()")
        return self._candidate.run(samples)

    def complete_switch(self) -> bool:
        """候选环路已稳定(locked)时完成交接并返回 True;
        未稳定时返回 False,继续跟踪原参考,状态不变。

        交接语义:新在用环路 omega 取候选环路的收敛估计,
        theta 取交接前在用环路的当前相位,输出相位连续;
        两路参考的真实相位差由环路按正常动态逐渐拉入,
        不在交接瞬间跳变,也不累积人为偏移。
        """
        if self._candidate is None:
            raise PLLError("complete_switch: 尚未调用 begin_switch()")
        if not self._candidate.locked:
            return False
        self._active = PLL(
            self._active.params,
            theta0=self._active.theta,
            omega0=self._candidate.omega,
        )
        self._candidate = None
        self._switch_count += 1
        return True

    def abort_switch(self) -> None:
        """放弃本次切换,丢弃候选环路,继续跟踪原参考。"""
        self._candidate = None
