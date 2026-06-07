# src/adaptive_engine.py
"""
Hybrid ALS + IRT recommendation engine.

All inference runs in numpy — no Spark required at recommendation time.

Score per candidate:
    combined = α × norm(ALS dot product)  +  β × norm(–|θ_user – b_task|)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

import src.data_config as config
from src.irt_model import IRTAbilityTracker


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


class AdaptiveEngine:
    """
    Loads pre-trained ALS factors + IRT task parameters and serves fast
    adaptive recommendations without a live Spark session.
    """

    def __init__(
        self,
        user_factors_path: str   = config.USER_FACTORS_PATH,
        item_factors_path: str   = config.ITEM_FACTORS_PATH,
        irt_params_path:   str   = config.IRT_PARAMS_PATH,
        als_weight:        float = config.ALS_WEIGHT,
        irt_weight:        float = config.IRT_WEIGHT,
        n_candidates:      int   = config.N_CANDIDATES,
    ):
        self.als_weight   = als_weight
        self.irt_weight   = irt_weight
        self.n_candidates = n_candidates
        self._load_als_factors(user_factors_path, item_factors_path)
        self._load_irt_params(irt_params_path)


    def _load_als_factors(self, user_path: str, item_path: str) -> None:
        u = np.load(user_path)
        v = np.load(item_path)
        self._user_ids     = u["ids"]
        self._user_factors = u["factors"].astype(np.float32)
        self._item_ids     = v["ids"]
        self._item_factors = v["factors"].astype(np.float32)
        self._user_idx     = {int(uid): i for i, uid in enumerate(self._user_ids)}
        self._item_idx     = {int(iid): i for i, iid in enumerate(self._item_ids)}
        print(f"[Engine] {len(self._user_ids)} users, {len(self._item_ids)} items, "
              f"rank={self._user_factors.shape[1]}")

    def _load_irt_params(self, irt_path: str) -> None:
        self._irt = pd.read_parquet(irt_path).set_index("task_id")
        print(f"[Engine] IRT params for {len(self._irt)} tasks")

    def known_user(self, user_id: int) -> bool:
        return user_id in self._user_idx

    def recommend(
        self,
        user_id:         int,
        ability_tracker: IRTAbilityTracker,
        seen_task_ids:   Set[int],
        topic_filter:    Optional[str] = None,
        n_return:        int = 1,
    ) -> List[Dict]:
        """
        1. ALS dot product scores for all items
        2. Filter by topic / exclude seen
        3. Keep top-N_CANDIDATES by ALS score
        4. Re-rank: combined = α·norm(ALS) + β·norm(–|θ–b|)
        5. Return top-n_return as dicts
        """
        # ALS scores
        if self.known_user(user_id):
            u_vec = self._user_factors[self._user_idx[user_id]]
        else:
            u_vec = self._user_factors.mean(axis=0)   # cold start
        als_scores = self._item_factors @ u_vec

        cands = pd.DataFrame({"task_id": self._item_ids, "als_score": als_scores})
        cands = cands[~cands["task_id"].isin(seen_task_ids)]

        if topic_filter:
            valid = self._irt[self._irt["topic_group"] == topic_filter].index
            cands = cands[cands["task_id"].isin(valid)]

        if cands.empty:
            return []

        # ALS pre-selection
        cands = cands.nlargest(min(self.n_candidates, len(cands)), "als_score")

        # Attach IRT metadata
        cands = cands.join(
            self._irt[["difficulty_irt", "primary_skill", "secondary_skill", "topic_group"]],
            on="task_id", how="left",
        ).dropna(subset=["difficulty_irt"])

        # IRT difficulty-match score: maximised when θ ≈ b
        def irt_match(row) -> float:
            theta = ability_tracker.effective_ability(
                int(row["primary_skill"]),
                int(row.get("secondary_skill", 0) or 0),
            )
            return -abs(theta - float(row["difficulty_irt"]))

        cands["irt_score"] = cands.apply(irt_match, axis=1)

        # Combined score
        cands["als_norm"] = _minmax(cands["als_score"].to_numpy())
        cands["irt_norm"] = _minmax(cands["irt_score"].to_numpy())
        cands["score"]    = self.als_weight * cands["als_norm"] + self.irt_weight * cands["irt_norm"]

        top = cands.nlargest(n_return, "score")
        return [
            {
                "task_id":         int(row["task_id"]),
                "topic_group":     row.get("topic_group", "?"),
                "primary_skill":   int(row["primary_skill"]),
                "secondary_skill": int(row.get("secondary_skill", 0) or 0),
                "difficulty_irt":  round(float(row["difficulty_irt"]), 3),
                "als_score":       round(float(row["als_score"]), 4),
                "irt_score":       round(float(row["irt_score"]), 4),
                "combined_score":  round(float(row["score"]), 4),
            }
            for _, row in top.iterrows()
        ]
