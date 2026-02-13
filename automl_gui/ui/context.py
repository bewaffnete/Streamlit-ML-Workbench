"""UI context object for dependency injection into tab renderers."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import RuntimeConfig
from ..core.jobs import JobManager
from ..services import ServiceContainer
from ..state import AppState


@dataclass(slots=True)
class UIContext:
    services: ServiceContainer
    state: AppState
    runtime_config: RuntimeConfig
    job_manager: JobManager
