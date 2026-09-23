"""整数采样数字锁相环(PLL)捕获诊断模块。

基于合成复数导频的二阶离散 PLL。每个有效样本按如下顺序更新:

1. 旋转:   y = z * exp(-j*theta)
2. 误差:   error = wrap(atan2(Im y, Re y))  ∈ [-pi, pi),单位 rad
3. 频率:   omega' = clip(omega + ki*error, -wmax, +wmax)
   (omega 与 wmax 的单位均为 rad/sample,即每个采样间隔的相位增量)
4. 相位:   theta' = wrap(theta + omega' + kp*error),theta 单位 rad

物理量换算(采样周期 Ts 单位为秒, 采样率 fs = 1/Ts 单位 Hz):

- 物理频偏 f(Hz) = omega / (2*pi*Ts) = omega * fs / (2*pi)
- 信号在采样点上的相位模型: phi_k = phi0 + 2*pi*f*k*Ts = phi0 + omega*k

运行中可通过 PLL.set_sample_period() 切换采样周期(见该方法文档):
theta 保持不变(切换瞬间物理相位连续),omega/wmax 按 r=Ts_new/Ts_old 线性
缩放,增益按 kp'=r*kp、ki'=r^2*ki 缩放以保持物理时间域环路动态(自然频率
omega_n=sqrt(ki)/Ts 与阻尼 zeta=kp/(2*sqrt(ki)))不变。

幅度 < amin 的样本视为无效:冻结 omega/theta 更新、清零捕获计数、
计一次失锁并在诊断中标记 valid=False。

连续 lock_count 个有效样本满足 |error| <= eps 时进入 locked;
任一有效样本 |error| > eps 立即清零计数并退出 locked。

仅依赖 Python 标准库(math / dataclasses)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

__all__ = [
    "PLLParams",
    "PLL",
    "Diagnostic",
    "PLLResult",
    "RateSwitchReport",
    "SamplePeriod",
    "PLLError",
    "PLLParamError",
    "PLLInputError",
    "PLLSwitchError",
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


class PLLSwitchError(PLLError):
    """运行中切换采样周期的请求非法(新周期非有限正数等)。"""


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
class SamplePeriod:
    """采样周期的单位封装。

    period: 采样间隔的数值。
    unit:   "seconds"(秒)或 "samples"(无量纲样本, period 必须 == 1.0)。
    不提供独立的 fs/Hz 构造器:Hz = 1/seconds, 用 SamplePeriod.seconds(1/fs) 即可。
    """

    period: float = 1.0
    unit: str = "seconds"

    def __post_init__(self) -> None:
        if self.unit not in ("seconds", "samples"):
            raise PLLParamError(
                f"unit: 需要 'seconds' 或 'samples', 得到 {self.unit!r}"
            )
        period = _finite_real("period", self.period, PLLParamError)
        if period <= 0.0:
            raise PLLParamError(f"period: 需要 > 0, 得到 {period!r}")
        if self.unit == "samples" and period != 1.0:
            raise PLLParamError(
                f"period: unit='samples' 时必须为 1.0(每样本就是一个单位), "
                f"得到 {period!r}"
            )
        object.__setattr__(self, "period", period)

    @classmethod
    def seconds(cls, period: float) -> "SamplePeriod":
        """以秒为单位的采样周期 Ts(秒)。"""
        return cls(period, "seconds")

    def to_seconds(self) -> float:
        """返回以秒计的采样周期;无量纲单位视为 1 秒。"""
        return self.period if self.unit == "seconds" else 1.0


@dataclass(frozen=True)
class PLLParams:
    """环路参数。构造时校验稳定范围: 0 < kp < 1 且 0 < ki < kp。

    kp:         比例增益(相位步进),必须满足 0 < kp < 1。
    ki:         积分增益(频率步进),必须满足 0 < ki < kp。
    wmax:       频率字限幅幅度(rad/sample),有限且 > 0。
    amin:       有效样本最小幅度,有限且 >= 0。
    eps:        捕获误差门限(rad),有限且 > 0。
    lock_count: 进入 locked 所需的连续有效样本数 L,int 且 >= 1(拒绝 bool)。
    ts:         采样周期(SamplePeriod,默认 SamplePeriod() 即 1 秒/单位样本)。
                仅用于物理量换算与切换时的增益/频率缩放,逐样本更新方程不变。
    """

    kp: float
    ki: float
    wmax: float
    amin: float
    eps: float
    lock_count: int
    ts: SamplePeriod = SamplePeriod()

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
        ts = self.ts
        if not isinstance(ts, SamplePeriod):
            raise PLLParamError(
                f"ts: 需要 SamplePeriod, 得到 {type(ts).__name__}({ts!r})"
            )
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


@dataclass(frozen=True)
class RateSwitchReport:
    """一次采样周期切换的报告(set_sample_period 的返回值)。

    index:             切换发生在第几个样本之后(== 当时的 processed)。
    period_old/new:    切换前/后的采样周期(秒)。
    ratio:             r = period_new / period_old。
    params_old/new:    切换前/后的环路参数(新参数为 kp*r、ki*r^2、wmax*r)。
    theta_old/new:     切换前/后相位(rad),保持物理相位连续故数值相同。
    omega_old/new:     切换前/后频率字(rad/sample),new = old * r。
    physical_frequency_hz: 切换前后保持不变的物理频偏(Hz)。
    """

    index: int
    period_old: float
    period_new: float
    ratio: float
    params_old: PLLParams
    params_new: PLLParams
    theta_old: float
    theta_new: float
    omega_old: float
    omega_new: float
    physical_frequency_hz: float


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
        #: 时间基准周期(秒),默认等于构造参数 ts;set_sample_period() 时更新。
        self._ts = params.ts.to_seconds()

    # ---- 只读状态 ----
    @property
    def params(self) -> PLLParams:
        return self._params

    @property
    def sample_period(self) -> SamplePeriod:
        """当前采样周期(与 params.ts 同步)。"""
        return self._params.ts

    @property
    def sample_period_seconds(self) -> float:
        """当前采样周期的秒数(无量纲单位视为 1 秒)。"""
        return self._ts

    @property
    def sample_rate(self) -> float:
        """当前采样率 fs = 1/Ts(Hz)。"""
        return 1.0 / self._ts

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def frequency_hz(self) -> float:
        """当前估计的物理频偏(Hz): omega / (2*pi*Ts)。"""
        return self._omega / (_TWO_PI * self._ts)

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

    # ---- 运行中切换采样周期 ----
    def set_sample_period(self, ts) -> RateSwitchReport:
        """在样本之间切换采样周期,保持物理相位连续与物理时间域环路动态不变。

        ts: 新采样周期(SamplePeriod;或(有限正数, 单位字符串)二元组/
            单个有限正数——后者按秒处理)。

        约定:切换在"上一个样本之后、下一个样本之前"生效,下一个样本与上一个
        样本的物理间隔为一个新周期。设 r = Ts_new/Ts_old,则一次性原子地:

        - theta_new = theta_old(切换瞬间物理相位连续,数值不变,单位 rad);
        - omega_new = r * omega_old(rad/sample 随周期缩放,物理频偏 Hz 不变);
        - kp_new = r*kp, ki_new = r^2*ki, wmax_new = r*wmax
          (保持 omega_n = sqrt(ki)/Ts 与 zeta = kp/(2*sqrt(ki)) 不变;
           若 omega 已限幅,|omega_new| <= wmax_new 仍成立);
        - eps/amin/lock_count 不变(相位量与样本计数,与采样周期无关);
        - streak/locked/lost_samples/processed 均不变(切换不消耗样本)。

        非法请求(NaN/无穷/<=0、单位字符串非法等)抛 PLLSwitchError,
        所有状态保持调用前原样。
        """
        # ---- 入站校验全部先于任何状态变更 ----
        try:
            if isinstance(ts, SamplePeriod):
                new_sp = ts
            elif isinstance(ts, tuple) and len(ts) == 2:
                new_sp = SamplePeriod(ts[0], ts[1])
            elif isinstance(ts, tuple):
                raise PLLSwitchError(
                    f"ts: 元组形式必须为 (period, unit) 且长度为 2, 得到长度 {len(ts)}"
                )
            else:
                # 单个有限正数按秒处理。
                new_sp = SamplePeriod(ts, "seconds")
        except PLLParamError as e:
            # SamplePeriod 的构造错误统一为本方法的切换错误。
            raise PLLSwitchError(f"ts: 非法采样周期: {e}") from None
        ts_new = new_sp.to_seconds()
        r = ts_new / self._ts
        if not math.isfinite(r) or r <= 0.0:
            # 理论上两个有限正周期之比必为有限正数,此处为防御性兜底。
            raise PLLSwitchError(
                f"ratio: Ts_new/Ts_old={r!r} 非有限正数"
            )

        p_old = self._params
        kp_new = p_old.kp * r
        ki_new = p_old.ki * r * r
        wmax_new = p_old.wmax * r
        omega_old = self._omega
        omega_new = omega_old * r
        theta_old = self._theta
        params_old = p_old
        period_old = self._ts

        # 新参数(稳定性范围 0<kp<1 可能在 r 很大时被破坏——显式拒绝,
        # 因为那意味着新采样率下逐样本比例增益不稳定)。
        if not 0.0 < kp_new < 1.0:
            raise PLLSwitchError(
                f"切换后 kp'={kp_new!r}(=kp*r, r={r!r}) 不满足 0 < kp' < 1; "
                "拒绝产生不稳定环路,请选择更接近当前周期的值"
            )
        if not 0.0 < ki_new < kp_new:
            # r>0 时 ki_new/kp_new = (ki/kp)*r, 仅在 r 极大时可能越界。
            raise PLLSwitchError(
                f"切换后 ki'={ki_new!r} 不满足 0 < ki' < kp'={kp_new!r} "
                f"(r={r!r});拒绝产生不稳定环路"
            )

        # ---- 原子提交 ----
        p_new = replace(
            p_old, kp=kp_new, ki=ki_new, wmax=wmax_new, ts=new_sp
        )
        self._params = p_new
        self._ts = ts_new
        self._omega = omega_new
        # theta 数值不变(wrap 后),无需改写;明确不改 streak/lost/processed。
        return RateSwitchReport(
            index=self._processed,
            period_old=period_old,
            period_new=ts_new,
            ratio=r,
            params_old=params_old,
            params_new=p_new,
            theta_old=theta_old,
            theta_new=theta_old,
            omega_old=omega_old,
            omega_new=omega_new,
            physical_frequency_hz=omega_old / (_TWO_PI * period_old),
        )

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
