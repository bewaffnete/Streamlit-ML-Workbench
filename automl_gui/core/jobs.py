"""Background job manager for non-blocking training workloads."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import uuid
from typing import Any

from ..config import RuntimeConfig, TrainConfig
from ..evaluation import MetricsService
from ..logging_utils import get_logger
from ..models import build_default_registry
from ..preprocessing import (
    DataCleaningRequest,
    PreprocessorConfig,
    PreprocessorFactory,
)
from ..services import TrainingService
from ..training import ModelTrainer, RandomizedSearchOptimizer

LOGGER = get_logger("jobs")

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None


@dataclass(slots=True)
class TrainingJobPayload:
    df: Any
    task_type: str
    target_col: str | None
    selected_features: list[str]
    cleaning_request: dict[str, Any]
    preprocessor_config: dict[str, Any]
    train_config: dict[str, Any]
    dataset_fingerprint: str | None
    run_permutation_importance: bool
    runtime_config: dict[str, Any]


def _training_worker(payload: TrainingJobPayload):
    runtime_cfg = RuntimeConfig(**payload.runtime_config)
    train_cfg = TrainConfig(**payload.train_config)
    train_cfg.task_type = payload.task_type

    pre_cfg = PreprocessorConfig.from_dict(payload.preprocessor_config)
    cleaning_req = DataCleaningRequest(**payload.cleaning_request)

    registry = build_default_registry(
        n_clusters=train_cfg.n_clusters,
        max_n_jobs=runtime_cfg.max_n_jobs,
    )
    trainer = ModelTrainer(
        optimizer=RandomizedSearchOptimizer(n_jobs=runtime_cfg.max_n_jobs),
        n_jobs=runtime_cfg.max_n_jobs,
    )
    training_service = TrainingService(
        model_registry=registry,
        preprocessor_factory=PreprocessorFactory(),
        trainer=trainer,
        metrics_service=MetricsService(),
    )
    prepared = training_service.prepare_training_data(
        df=payload.df,
        task_type=payload.task_type,
        target_col=payload.target_col,
        selected_features=payload.selected_features,
        cleaning_request=cleaning_req,
    )
    return training_service.train_models(
        prepared_data=prepared,
        preprocessor_config=pre_cfg,
        train_config=train_cfg,
        dataset_fingerprint=payload.dataset_fingerprint,
        run_permutation_importance=payload.run_permutation_importance,
    )


class JobManager:
    """Process-pool based background job manager."""

    def __init__(self, *, max_workers: int = 1, prefer_processes: bool = True) -> None:
        self._executor_kind = "process"
        if prefer_processes:
            try:
                self._executor = ProcessPoolExecutor(max_workers=max_workers)
            except (PermissionError, OSError, ValueError) as exc:
                LOGGER.warning(
                    "ProcessPool unavailable; using ThreadPoolExecutor fallback: %s",
                    exc,
                    exc_info=True,
                )
                self._executor = ThreadPoolExecutor(max_workers=max_workers)
                self._executor_kind = "thread"
        else:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            self._executor_kind = "thread"
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit_training(self, payload: TrainingJobPayload) -> str:
        job_id = uuid.uuid4().hex
        future = self._executor.submit(_training_worker, payload)
        with self._lock:
            self._futures[job_id] = future
        LOGGER.info("Submitted training job %s via %s executor", job_id, self._executor_kind)
        return job_id

    def status(self, job_id: str) -> str:
        future = self._futures.get(job_id)
        if future is None:
            return "unknown"
        if future.running():
            return "running"
        if future.cancelled():
            return "cancelled"
        if future.done():
            return "failed" if future.exception() else "completed"
        return "queued"

    def result(self, job_id: str):
        future = self._futures.get(job_id)
        if future is None:
            raise KeyError(f"Unknown job id: {job_id}")
        return future.result()

    def error(self, job_id: str) -> str | None:
        future = self._futures.get(job_id)
        if future is None or not future.done():
            return None
        exc = future.exception()
        return str(exc) if exc else None

    def cancel(self, job_id: str) -> bool:
        future = self._futures.get(job_id)
        if future is None:
            return False
        cancelled = future.cancel()
        if cancelled:
            LOGGER.info("Cancelled job %s", job_id)
        return cancelled

    def cleanup(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)


if st is not None:
    _cache_resource = st.cache_resource
else:  # pragma: no cover
    def _cache_resource(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        return _decorator


@_cache_resource(show_spinner=False)
def get_job_manager(max_workers: int = 1, prefer_processes: bool = True) -> JobManager:
    """Cached singleton JobManager for Streamlit session."""
    return JobManager(max_workers=max_workers, prefer_processes=prefer_processes)
