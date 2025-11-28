from .actions import IR, Action
from .compute import ApplyComputeConfig
from .data import ApplyChatFormat, ApplyQAFormat
from .defaults import ApplyDefaults
from .train import (
    ApplyDistributedTraining,
    ApplyFastKernelsOptimization,
    ApplyGradientCheckpointing,
    ApplyLoRAConfig,
    ApplyMoEOptimization,
    ApplyOptimalBatchSize,
    ApplyTrainingOptimization,
)

ACTIONS = [
    ApplyDefaults,
    ApplyComputeConfig,
    ApplyTrainingOptimization,
    ApplyDistributedTraining,
    ApplyFastKernelsOptimization,
    ApplyGradientCheckpointing,
    ApplyLoRAConfig,
    ApplyMoEOptimization,
    ApplyOptimalBatchSize,
    ApplyChatFormat,
    ApplyQAFormat,
]
