"""mech_spoof — mechanistic interpretability of instruction privilege in LLMs."""

from mech_spoof.configs import MODEL_CONFIGS, ModelConfig, DATA_DIR, RESULTS_DIR

__version__ = "0.1.0"

__all__ = [
    "MODEL_CONFIGS",
    "ModelConfig",
    "DATA_DIR",
    "RESULTS_DIR",
    "__version__",
]
