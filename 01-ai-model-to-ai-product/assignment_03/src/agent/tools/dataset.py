"""Bitext Customer Service dataset loader (cached pandas DataFrame)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def load_bitext() -> pd.DataFrame:
    """Load the Bitext customer-service dataset as a cached pandas DataFrame.

    The result is memoized for the process lifetime, so repeated tool calls
    reuse a single in-memory copy. We download anonymously (no HF token needed),
    silencing the routine "unauthenticated requests" notice so it doesn't clutter
    the agent's reasoning trace.
    """
    import logging

    from datasets import load_dataset
    from datasets.utils import logging as ds_logging

    # Quiet the harmless "set a HF_TOKEN" notice emitted on anonymous downloads.
    ds_logging.set_verbosity_error()
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
    )
    return ds.to_pandas()
