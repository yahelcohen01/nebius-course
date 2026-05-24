"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str
    nebius_base_url: str
    model_router: str
    model_agent: str
    checkpoint_db: str
    profile_dir: str
    max_iterations: int


def load_settings() -> Settings:
    return Settings(
        nebius_api_key=os.environ.get("NEBIUS_API_KEY", ""),
        nebius_base_url=os.environ.get(
            "NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"
        ),
        model_router=os.environ.get("MODEL_ROUTER", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        model_agent=os.environ.get("MODEL_AGENT", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
        checkpoint_db=os.environ.get("CHECKPOINT_DB", "data/checkpoints.sqlite"),
        profile_dir=os.environ.get("PROFILE_DIR", "data/profiles"),
        max_iterations=int(os.environ.get("MAX_ITERATIONS", "12")),
    )
