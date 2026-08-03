"""Thr_DT Decision Transformer model (inference only) — vendored verbatim
except for the import lines. See ../README.md."""
from .dt_config import ModelConfig
from .decision_transformer import build_model, count_parameters

__all__ = ["ModelConfig", "build_model", "count_parameters"]
