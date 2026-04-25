"""Per-experiment pipelines composed from library primitives."""

from mech_spoof.experiments.exp1_authority import run_experiment_1
from mech_spoof.experiments.exp1b_authority_conflict import run_experiment_1b
from mech_spoof.experiments.exp2_conflict import run_experiment_2
from mech_spoof.experiments.exp2b_conflict_evolved import run_experiment_2b
from mech_spoof.experiments.exp3_refusal import run_experiment_3
from mech_spoof.experiments.exp4_attacks import run_experiment_4
from mech_spoof.experiments.exp5_comparative import aggregate_results
from mech_spoof.experiments.rescore_positions import rescore_exp2b_at_all_positions

__all__ = [
    "aggregate_results",
    "run_experiment_1",
    "run_experiment_1b",
    "run_experiment_2",
    "run_experiment_2b",
    "run_experiment_3",
    "run_experiment_4",
    "rescore_exp2b_at_all_positions",
]
