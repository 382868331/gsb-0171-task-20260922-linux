"""整数采样数字锁相环（DPLL）捕获诊断可复用模块。

本包实现题面契约规定的二阶离散锁相环，跟踪合成复数导频的载波相位与
（每样本弧度的）频偏，并对每个样本输出逐样本诊断与失锁/锁定状态。

仅依赖 Python 3.14 标准库（math / cmath）。
"""

from .model import (
    PLLDiagnosis,
    PLLParams,
    PLLSample,
    PLLState,
)
from .core import (
    PLLConfigError,
    PLLInputError,
    PLL,
    track,
)

__all__ = [
    "PLL",
    "PLLParams",
    "PLLState",
    "PLLSample",
    "PLLDiagnosis",
    "PLLConfigError",
    "PLLInputError",
    "track",
]
