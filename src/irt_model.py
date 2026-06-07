# src/irt_model.py
"""
Item Response Theory — 1-Parameter Logistic (Rasch) model.

Offline:  calibrate_item_difficulties()  →  data-derived b_i per task
Online:   IRTAbilityTracker              →  per-skill θ updated after each answer

Rasch update rule
-----------------
  P(correct | θ, b) = sigmoid(SCALE × (θ – b))

  Correct:   θ ← θ + lr × (1 – P)   [large jump if task was hard]
  Incorrect: θ ← θ – lr × P          [large drop if task was easy]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

import src.data_config as config

_RASCH_SCALE = 3.5   # discrimination parameter (fixed in 1PL)


# helpers 

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def rasch_p_correct(theta: float, b: float) -> float:
    """P(correct) under the Rasch model.  θ and b are both ∈ [0, 1]."""
    return _sigmoid(_RASCH_SCALE * (theta - b))


# Online ability tracker

@dataclass
class IRTAbilityTracker:
    """
    Tracks per-skill ability θ ∈ [0.01, 0.99] during a live session.

    Parameters
    ----------
    initial_abilities : dict  {skill_id → θ}
        Warm-start values for returning users.  New users get DEFAULT_ABILITY.
    learning_rate : float
        Controls how fast ability changes after each answer.
    """
    initial_abilities: Optional[Dict[int, float]] = None
    learning_rate: float = config.IRT_LEARNING_RATE

    def __post_init__(self):
        self._abilities: Dict[int, float] = dict(self.initial_abilities or {})

    def get_ability(self, skill_id: int) -> float:
        return self._abilities.get(skill_id, config.DEFAULT_ABILITY)

    def effective_ability(self, primary_skill: int, secondary_skill: int = 0) -> float:
        """Weighted blend of primary + optional secondary skill abilities."""
        theta = self.get_ability(primary_skill)
        if secondary_skill and secondary_skill != 0:
            w = config.SECONDARY_SKILL_WEIGHT
            theta = (1.0 - w) * theta + w * self.get_ability(secondary_skill)
        return theta

    def update(
        self,
        primary_skill: int,
        task_difficulty_irt: float,
        is_correct: bool,
        secondary_skill: int = 0,
    ) -> Dict[int, float]:
        """Apply Rasch update after one answered exercise. Returns updated skills."""
        b     = float(np.clip(task_difficulty_irt, 0.01, 0.99))
        theta = self.effective_ability(primary_skill, secondary_skill)
        p     = rasch_p_correct(theta, b)

        delta = self.learning_rate * (1.0 - p) if is_correct else -self.learning_rate * p

        old_p = self.get_ability(primary_skill)
        new_p = float(np.clip(old_p + delta, 0.01, 0.99))
        self._abilities[primary_skill] = new_p
        updates = {primary_skill: new_p}

        if secondary_skill and secondary_skill != 0:
            old_s = self.get_ability(secondary_skill)
            new_s = float(np.clip(old_s + 0.6 * delta, 0.01, 0.99))
            self._abilities[secondary_skill] = new_s
            updates[secondary_skill] = new_s

        return updates

    @property
    def all_abilities(self) -> Dict[int, float]:
        return dict(self._abilities)

    def snapshot(self) -> Dict[int, float]:
        return dict(self._abilities)


# Offline calibration

def calibrate_item_difficulties(spark, interactions_path: str, tasks_path: str) -> pd.DataFrame:
    """
    Estimate IRT item difficulties from historical data (Spark required).

    Method: difficulty_irt = 1 – empirical_correct_rate
    Blend:  90 % empirical  +  10 % normalised simulation difficulty
            (regularises tasks with few observations)

    Returns a pandas DataFrame saved to config.IRT_PARAMS_PATH by the caller.
    """
    from pyspark.sql import functions as F

    interactions = spark.read.parquet(interactions_path)
    tasks        = spark.read.parquet(tasks_path)

    task_stats = (
        interactions
        .groupBy("task_id")
        .agg(
            F.avg("is_correct").alias("p_correct_empirical"),
            F.count("*").alias("n_interactions"),
        )
    )

    joined = (
        tasks
        .select("task_id", "difficulty", "primary_skill", "secondary_skill", "topic_group")
        .join(task_stats, on="task_id", how="left")
    )

    df = joined.toPandas()

    # Normalise simulation difficulty [1, 10] - [0.05, 0.95]
    df["difficulty_norm"] = ((df["difficulty"] - 1.0) / 9.0 * 0.90 + 0.05).clip(0.05, 0.95)

    p_hat = df["p_correct_empirical"].fillna(df["difficulty_norm"]).clip(0.05, 0.95)
    empirical_diff = (1.0 - p_hat).clip(0.05, 0.95)

    has_data = df["n_interactions"].fillna(0) >= 20
    df["difficulty_irt"] = np.where(
        has_data,
        (0.90 * empirical_diff + 0.10 * df["difficulty_norm"]).clip(0.05, 0.95),
        df["difficulty_norm"],
    )
    df["p_correct_empirical"] = p_hat.values
    df = df.rename(columns={"difficulty": "difficulty_raw"})

    return df[[
        "task_id", "primary_skill", "secondary_skill", "topic_group",
        "difficulty_raw", "difficulty_norm", "p_correct_empirical",
        "n_interactions", "difficulty_irt",
    ]]
