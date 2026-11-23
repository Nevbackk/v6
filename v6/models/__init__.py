"""V6 geometry-aware tokenizer, pairwise relations, and scalar/vector filter."""
from .feature_builder import FeatureBatch, FeatureBuilder
from .normalization import V6Normalization
from .geometry import FrameGeometry, build_frame_geometry
from .scalar_vector import ScalarVectorEncoder
from .tokenizer import Tokenizer, TokenizerOutput, V6Tokenizer
from .geometry_relation import build_pairwise_geometry, build_current_pairwise_geometry
from .scalar_vector_attention import RelationConditionedScalarVectorAttention
from .batch_time import BatchTimePlan, prepare_batch_time_plan
from .batched_physics import propagate_scaled_batch
from .geometry_filter import (GeometryConditionedFilter, GeometryFilterOutput,
    RecursiveFilterOutput, TrainingFilterOutput, V6GainHead, V6CorrectionHead,
    FilterDivergenceError)

__all__ = [
    "FeatureBatch",
    "FeatureBuilder",
    "V6Normalization",
    "FrameGeometry",
    "build_frame_geometry",
    "ScalarVectorEncoder",
    "Tokenizer",
    "TokenizerOutput",
    "V6Tokenizer",
    "build_pairwise_geometry",
    "build_current_pairwise_geometry",
    "RelationConditionedScalarVectorAttention",
    "BatchTimePlan",
    "prepare_batch_time_plan",
    "propagate_scaled_batch",
    "GeometryConditionedFilter",
    "GeometryFilterOutput",
    "RecursiveFilterOutput",
    "TrainingFilterOutput",
    "V6GainHead",
    "V6CorrectionHead",
    "FilterDivergenceError",
]
