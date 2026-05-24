"""Bitext Customer Service dataset loader (cached pandas DataFrame)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def load_bitext() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    return ds.to_pandas()
