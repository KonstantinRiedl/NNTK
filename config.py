from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 10**3
    beta: float = 0.52
    freeze_first_layer: bool = False

    n_points: int = 128
    epochs: int = 5000
    adam_epochs: Optional[int] = None  # defaults to 10 * epochs
    gd_epochs: Optional[int] = None  # defaults to 10 * epochs

    def __post_init__(self):
        if self.adam_epochs is None:
            self.adam_epochs = 10 * self.epochs
        if self.gd_epochs is None:
            self.gd_epochs = 10 * self.epochs

    gd_lr: float = 1e-2
    adam_lr: float = 1e-2
    damping0: float = 1e-8

    freq0: int = 2
    freq1: int = 8
    freq2: int = 10
    freq3: int = 16

    linear_coef: float = 4.0

    amp0: float = 1.0
    amp1: float = 0.8
    amp2: float = 0.8
    amp3: float = 0.8

    n_runs: int = 10

    # Set exactly one snapshot interval; the other must be None.
    iterate_interval: Optional[int] = None
    iterate_time_interval: Optional[float] = 0.4

    code: str = "FINAL"