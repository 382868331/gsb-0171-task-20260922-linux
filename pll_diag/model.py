"""公开数据结构：参数、状态、逐样本诊断与整段诊断结果。

角度约定：所有相位/频偏单位为弧度、弧度/样本，取值区间
[-pi, pi)（状态相位）与 [-wmax, wmax]（频偏估计）。

这些类型均为不可值（frozen dataclass）；有状态推进由
``pll_diag.core.PLL`` 负责，每步返回新的状态对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["PLLParams", "PLLState", "PLLSample", "PLLDiagnosis"]


@dataclass(frozen=True, slots=True)
class PLLParams:
    """PLL 环路参数。

    Attributes:
        kp: 比例增益，稳定范围要求 0 < kp < 1。
        ki: 积分增益，稳定范围要求 0 < ki < kp。
        wmax: 频偏积分器限幅（弧度/样本），必须为正有限值。
        eps: 锁定判定误差门限（弧度），必须为正有限值。
        L: 连续有效样本数阈值，正整数。
        amin: 有效样本最小幅度（严格小于它判失锁冻结），必须为正有限值。
    """

    kp: float
    ki: float
    wmax: float
    eps: float
    L: int
    amin: float = 1e-12


@dataclass(frozen=True, slots=True)
class PLLState:
    """PLL 在某一时刻的内部状态（处理完若干样本之后）。

    Attributes:
        theta: 当前相位估计（弧度），归一化到 [-pi, pi)。
        omega: 当前频偏估计（弧度/样本），限幅在 [-wmax, wmax]。
        streak: 已连续满足 ``abs(error) <= eps`` 的有效样本数；
            失锁样本或超差样本会立即清零。
        locked: ``streak >= L`` 是否成立。
        processed: 已处理样本总数（含失锁样本）。
        lost_count: 累计失锁（幅度不足）样本数。
    """

    theta: float
    omega: float
    streak: int
    locked: bool
    processed: int
    lost_count: int


@dataclass(frozen=True, slots=True)
class PLLSample:
    """单个输入样本的逐样本诊断。

    Attributes:
        index: 样本的*全局*下标，即该 PLL 自启动以来处理的第几个样本
            （等于处理前的 ``state.processed``）。这保证任意分块与一次性
            处理的逐样本诊断下标完全一致；从零启动时就是样本序号。
        amplitude: 输入样本幅度 ``abs(z)``。
        valid: 幅度是否达到 ``amin``（False 表示失锁冻结）。
        error: 相位误差（弧度，[-pi, pi)）；失锁样本为 None。
        theta: 该步更新后的相位估计（失锁样本为冻结前/当前值）。
        omega: 该步更新后的频偏估计（失锁样本为冻结值）。
        omega_saturated: 该步积分器是否被 wmax 限幅；失锁样本为 False。
        streak_after: 该步结束后的连续合格计数。
        locked_after: 该步结束后是否处于锁定态。
    """

    index: int
    amplitude: float
    valid: bool
    error: float | None
    theta: float
    omega: float
    omega_saturated: bool
    streak_after: int
    locked_after: bool


@dataclass(frozen=True, slots=True)
class PLLDiagnosis:
    """一段样本处理完成后的完整诊断结果。

    Attributes:
        samples: 逐样本诊断，顺序与输入一致。
        final_state: 处理完本段后的 PLL 状态（等价于驱动该段的 PLL 状态）。
        locked: 终态是否锁定。
        locked_at: 本段内首次进入锁定时的样本*全局下标*；本段未锁定为 None。
            跨分块时，每段独立报告本段内首次锁定时刻（可能因失锁-重捕
            多次出现）。
        lost_indices: 本段内判为失锁的样本下标。
        saturated_indices: 本段内积分器发生限幅的样本下标。
        max_abs_error: 本段有效样本的最大绝对相位误差；无有效样本为 None。
    """

    samples: tuple[PLLSample, ...] = field(default_factory=tuple)
    final_state: PLLState = field(
        default_factory=lambda: PLLState(0.0, 0.0, 0, False, 0, 0)
    )
    locked: bool = False
    locked_at: int | None = None
    lost_indices: tuple[int, ...] = field(default_factory=tuple)
    saturated_indices: tuple[int, ...] = field(default_factory=tuple)
    max_abs_error: float | None = None

    @property
    def frequency_offset(self) -> float:
        """终态频偏估计（弧度/样本）。"""
        return self.final_state.omega

    @property
    def phase(self) -> float:
        """终态相位估计（弧度，[-pi, pi)）。"""
        return self.final_state.theta

    @property
    def status(self) -> Literal["locked", "unlocked"]:
        """终态状态：``"locked"`` 或 ``"unlocked"``。"""
        return "locked" if self.locked else "unlocked"
