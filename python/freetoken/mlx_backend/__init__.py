"""Apple-silicon (MLX) execution backend.

Replaces the CUDA scheduler process with an MLX one that speaks the identical ZMQ
backend protocol (UserMsg in, DetokenizeMsg/PromptAdmittedMsg/ErrorReplyMsg out), so
the API server, tokenizer workers, shell and all client-facing behavior are shared
between the two backends. Model execution is delegated to mlx-lm's model zoo.
"""

from .worker import MlxScheduler, mlx_scheduler_worker

__all__ = ["MlxScheduler", "mlx_scheduler_worker"]
