"""二阶离散锁相环核心：参数/输入校验、状态机与一次性便捷接口。

每步更新契约（对复数样本 z，当前相位估计 theta、频偏估计 omega）::

    r = z * exp(-j*theta)            # 先按 theta 旋转
    error = atan2(r) ∈ [-pi, pi)
    omega_unc = omega + ki * error
    omega' = clip(omega_unc, ±wmax)
    theta' = wrap(theta + omega' + kp * error) ∈ [-pi, pi)

幅度 ``abs(z) < amin`` 时：冻结 theta/omega 不更新，该样本计入失锁，
连续合格计数清零。对有效样本：``abs(error) <= eps`` 时计数 +1，否则
立即清零；计数达到 L 起判为 locked。

分块一致性：``PLL`` 的 ``update`` 按顺序消费分块，其结果与把所有分块
拼接后一次性 ``track`` 的逐样本结果、终态完全一致（同样的浮点运算、
同样的顺序）。校验在任何状态变更前完成，失败不留下部分变更。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .model import PLLDiagnosis, PLLParams, PLLSample, PLLState

__all__ = [
    "PLL",
    "track",
    "PLLConfigError",
    "PLLInputError",
    "wrap_angle",
    "initial_state",
    "MAX_SAMPLES_PER_CALL",
    "MAX_COMPONENT_MAGNITUDE",
    "MAX_L",
]

_TWO_PI = 2.0 * math.pi

#: 接口边界：单次 update 允许的最大样本数（README 同步声明）。
MAX_SAMPLES_PER_CALL = 10_000_000
#: 接口边界：样本实部/虚部允许的最大绝对值（保证旋转中间量不溢出）。
MAX_COMPONENT_MAGNITUDE = 1e100
#: 接口边界：连续锁定计数阈值 L 的上限。
MAX_L = 2_147_483_647


class PLLConfigError(ValueError):
    """环路参数不满足契约（稳定范围、正数、非 NaN/Infinity 等）。"""


class PLLInputError(ValueError):
    """输入样本或初始状态不满足契约（NaN/Infinity、bool 当整数等）。"""


def wrap_angle(a: float) -> float:
    """把角度归一化到 [-pi, pi)；拒绝 NaN/Infinity。"""
    _check_finite(a, "angle")
    w = (a + math.pi) % _TWO_PI - math.pi
    # 对模运算在负侧可能产生的 -pi? 不会：余数范围 [0,2pi)，再减 pi 得 [-pi,pi)。
    return w


def _check_finite(value: float, what: str) -> None:
    if isinstance(value, bool):
        raise PLLInputError(f"{what} 不能是 bool 类型")
    if not isinstance(value, (int, float)):
        raise PLLInputError(f"{what} 必须是实数，得到 {type(value).__name__}")
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise PLLInputError(f"{what} 拒绝 NaN/Infinity，得到 {value!r}")


def _check_positive_int(value: int, what: str) -> int:
    if isinstance(value, bool):
        raise PLLConfigError(f"{what} 不能是 bool 类型")
    if not isinstance(value, int):
        raise PLLConfigError(
            f"{what} 必须是整数 int，得到 {type(value).__name__}"
        )
    if value <= 0:
        raise PLLConfigError(f"{what} 必须为正整数，得到 {value}")
    return value


def validate_params(p: PLLParams) -> None:
    """按题面顺序校验参数；全部通过才返回，否则抛 PLLConfigError。"""
    if not isinstance(p, PLLParams):
        raise PLLConfigError(
            f"params 必须是 PLLParams，得到 {type(p).__name__}"
        )
    # 增益必须是有限实数（bool 也是 int，显式拒绝）。
    for name in ("kp", "ki", "wmax", "eps", "amin"):
        value = getattr(p, name)
        if isinstance(value, bool):
            raise PLLConfigError(f"参数 {name} 不能是 bool 类型")
        if not isinstance(value, (int, float)):
            raise PLLConfigError(
                f"参数 {name} 必须是数值，得到 {type(value).__name__}"
            )
        fv = float(value)
        if math.isnan(fv) or math.isinf(fv):
            raise PLLConfigError(f"参数 {name} 拒绝 NaN/Infinity，得到 {value!r}")

    _check_positive_int(p.L, "参数 L")
    if p.L > MAX_L:
        raise PLLConfigError(f"参数 L 超过上限 {MAX_L}，得到 {p.L}")

    kp, ki = float(p.kp), float(p.ki)
    # 题面稳定范围优先，且按此固定顺序报错，便于定位。
    if not (0.0 < kp < 1.0):
        raise PLLConfigError(
            f"稳定范围要求 0 < kp < 1，得到 kp={kp!r}"
        )
    if not (0.0 < ki < kp):
        raise PLLConfigError(
            f"稳定范围要求 0 < ki < kp（kp={kp!r}），得到 ki={ki!r}"
        )
    if not (float(p.wmax) > 0.0):
        raise PLLConfigError(f"wmax 必须为正，得到 {p.wmax!r}")
    if not (float(p.eps) > 0.0):
        raise PLLConfigError(f"eps 必须为正，得到 {p.eps!r}")
    if not (float(p.amin) > 0.0):
        raise PLLConfigError(f"amin 必须为正，得到 {p.amin!r}")


def _coerce_complex(z: object, where: str) -> complex:
    """把单个样本转成 complex，并拒绝 NaN/Infinity 与非数值类型。"""
    if isinstance(z, bool):
        raise PLLInputError(f"{where} 不能是 bool 类型")
    if isinstance(z, complex):
        c = z
    elif isinstance(z, (int, float)):
        c = complex(z)
    else:
        raise PLLInputError(
            f"{where} 必须是 complex/int/float，得到 {type(z).__name__}"
        )
    if math.isnan(c.real) or math.isnan(c.imag):
        raise PLLInputError(f"{where} 拒绝 NaN：{z!r}")
    if math.isinf(c.real) or math.isinf(c.imag):
        raise PLLInputError(f"{where} 拒绝 Infinity：{z!r}")
    if (
        abs(c.real) > MAX_COMPONENT_MAGNITUDE
        or abs(c.imag) > MAX_COMPONENT_MAGNITUDE
    ):
        raise PLLInputError(
            f"{where} 实部/虚部绝对值超过接口上限 "
            f"{MAX_COMPONENT_MAGNITUDE:g}：{z!r}"
        )
    return c


def initial_state() -> PLLState:
    """题面默认初始状态：theta=omega=0，未锁定。"""
    return PLLState(theta=0.0, omega=0.0, streak=0,
                    locked=False, processed=0, lost_count=0)


def _check_initial_state(
    s: PLLState, wmax: float | None = None, L: int | None = None
) -> PLLState:
    if not isinstance(s, PLLState):
        raise PLLInputError(
            f"initial_state 必须是 PLLState，得到 {type(s).__name__}"
        )
    _check_finite(s.theta, "initial_state.theta")
    _check_finite(s.omega, "initial_state.omega")
    if not (-math.pi <= s.theta < math.pi):
        raise PLLInputError(
            f"initial_state.theta 必须落在 [-pi, pi)，得到 {s.theta!r}"
        )
    if wmax is not None and not (-float(wmax) <= s.omega <= float(wmax)):
        raise PLLInputError(
            f"initial_state.omega 必须落在 [-wmax, wmax]="
            f"[{-float(wmax)}, {float(wmax)}]，得到 {s.omega!r}"
        )
    if isinstance(s.streak, bool) or not isinstance(s.streak, int) or s.streak < 0:
        raise PLLInputError("initial_state.streak 必须为非负整数")
    if not isinstance(s.locked, bool):
        raise PLLInputError("initial_state.locked 必须是 bool")
    if (
        isinstance(s.processed, bool)
        or not isinstance(s.processed, int)
        or s.processed < 0
    ):
        raise PLLInputError("initial_state.processed 必须为非负整数")
    if (
        isinstance(s.lost_count, bool)
        or not isinstance(s.lost_count, int)
        or s.lost_count < 0
    ):
        raise PLLInputError("initial_state.lost_count 必须为非负整数")
    if L is not None and bool(s.locked) != (s.streak >= L):
        raise PLLInputError(
            f"initial_state 不一致：locked={s.locked} 与 streak={s.streak}、"
            f"L={L} 矛盾（要求 locked == (streak >= L)）"
        )
    return s


class PLL:
    """有状态的二阶离散 PLL。

    典型用法::

        pll = PLL(PLLParams(kp=0.3, ki=0.05, wmax=0.5, eps=2e-2, L=8))
        diag1 = pll.update(block1)   # 第一批
        diag2 = pll.update(block2)   # 下一批，状态自动延续
        assert pll.state == diag2.final_state

    所有输入在任何状态变更前完成校验；``update`` 失败时对象状态保持为
    调用前快照（要么整段生效，要么完全不变）。
    """

    __slots__ = ("_p", "_state")

    def __init__(
        self,
        params: PLLParams,
        state: PLLState | None = None,
    ) -> None:
        validate_params(params)  # 先拒绝非法参数，再初始化任何状态
        self._p = params
        if state is not None:
            self._state = _check_initial_state(state, params.wmax, params.L)
        else:
            self._state = initial_state()

    @property
    def params(self) -> PLLParams:
        return self._p

    @property
    def state(self) -> PLLState:
        return self._state

    @property
    def locked(self) -> bool:
        return self._state.locked

    def reset(self, state: PLLState | None = None) -> None:
        """复位到给定状态（默认零状态）；校验失败时保持现状不变。"""
        if state is not None:
            new_state = _check_initial_state(state, self._p.wmax, self._p.L)
        else:
            new_state = initial_state()
        self._state = new_state

    def update(self, samples: Iterable[complex]) -> PLLDiagnosis:
        """消费一个分块的样本，返回该分块诊断，并推进内部状态。

        ``samples`` 必须可迭代且长度有限；先把整段物化为列表并逐个校验，
        任何一个样本非法则整体拒绝、内部状态不变。
        """
        if isinstance(samples, (str, bytes)):
            raise PLLInputError(
                f"samples 不能是 str/bytes，得到 {type(samples).__name__}"
            )
        # 已知长度时先于物化作长度检查，避免无意义地展开超大生成器。
        if hasattr(samples, "__len__"):
            try:
                n = len(samples)
            except TypeError:
                n = -1
            if n > MAX_SAMPLES_PER_CALL:
                raise PLLInputError(
                    f"单次 update 样本数超过上限 {MAX_SAMPLES_PER_CALL}，"
                    f"得到 {n}"
                )

        try:
            raw = list(samples)
        except TypeError as exc:
            raise PLLInputError("samples 必须是可迭代的样本序列") from exc

        if len(raw) > MAX_SAMPLES_PER_CALL:
            raise PLLInputError(
                f"单次 update 样本数超过上限 {MAX_SAMPLES_PER_CALL}，"
                f"得到 {len(raw)}"
            )

        # 先整段校验（定位到具体下标），再动状态——失败无部分变更。
        block = [
            _coerce_complex(z, f"samples[{i}]") for i, z in enumerate(raw)
        ]

        return self._advance(block)

    def _advance(self, block: Sequence[complex]) -> PLLDiagnosis:
        p = self._p
        kp, ki, wmax, eps, amin, L = (
            float(p.kp), float(p.ki), float(p.wmax),
            float(p.eps), float(p.amin), p.L,
        )
        st = self._state
        theta, omega = st.theta, st.omega
        streak, locked = st.streak, st.locked
        processed, lost_count = st.processed, st.lost_count

        out: list[PLLSample] = []
        lost_idx: list[int] = []
        sat_idx: list[int] = []
        locked_at: int | None = None
        max_err: float | None = None

        for z in block:
            gi = processed  # 全局样本下标（从零启动时即样本序号）
            amp = abs(z)
            if amp < amin:
                # 失锁：冻结状态，计失锁，清零连续计数。
                lost_count += 1
                streak = 0
                locked = False
                out.append(
                    PLLSample(
                        index=gi, amplitude=amp, valid=False, error=None,
                        theta=theta, omega=omega, omega_saturated=False,
                        streak_after=0, locked_after=False,
                    )
                )
                lost_idx.append(gi)
                processed += 1
                continue

            # 先按 theta 旋转样本：r = z * exp(-j theta)。
            c, s = math.cos(theta), math.sin(theta)
            r = complex(
                z.real * c + z.imag * s,
                z.imag * c - z.real * s,
            )
            error = math.atan2(r.imag, r.real)
            if error == math.pi:  # atan2 上界闭，统一到 [-pi, pi)
                error = -math.pi

            omega_unc = omega + ki * error
            omega_new = max(-wmax, min(wmax, omega_unc))
            saturated = omega_new != omega_unc

            theta_new = (theta + omega_new + kp * error + math.pi) % _TWO_PI - math.pi

            if abs(error) <= eps:
                streak += 1
            else:
                streak = 0
            locked = streak >= L
            if locked and locked_at is None:
                locked_at = gi

            if max_err is None or abs(error) > max_err:
                max_err = abs(error)
            if saturated:
                sat_idx.append(gi)

            theta, omega = theta_new, omega_new
            processed += 1
            out.append(
                PLLSample(
                    index=gi, amplitude=amp, valid=True, error=error,
                    theta=theta, omega=omega,
                    omega_saturated=saturated,
                    streak_after=streak, locked_after=locked,
                )
            )

        new_state = PLLState(
            theta=theta, omega=omega, streak=streak, locked=locked,
            processed=processed, lost_count=lost_count,
        )
        self._state = new_state
        return PLLDiagnosis(
            samples=tuple(out),
            final_state=new_state,
            locked=locked,
            locked_at=locked_at,
            lost_indices=tuple(lost_idx),
            saturated_indices=tuple(sat_idx),
            max_abs_error=max_err,
        )


def track(
    samples: Iterable[complex],
    params: PLLParams,
    initial: PLLState | None = None,
) -> PLLDiagnosis:
    """一次性处理整段样本的便捷接口（内部就是一个一次性 PLL）。"""
    return PLL(params, initial).update(samples)
