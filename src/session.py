# src/session.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

import src.data_config as config
from src.irt_model import IRTAbilityTracker, rasch_p_correct


@dataclass
class AnswerRecord:
    step:           int
    task_id:        int
    topic_group:    str
    primary_skill:  int
    difficulty_irt: float
    is_correct:     bool
    ability_before: float
    ability_after:  float
    p_correct:      float
    combined_score: float
    elapsed_sec:    float


class UserSession:
    def __init__(self, user_id: int, ability_tracker: IRTAbilityTracker):
        self.user_id        = user_id
        self.tracker        = ability_tracker
        self.seen:  Set[int]           = set()
        self.history: List[AnswerRecord] = []
        self._step          = 0
        self._start         = time.time()

    # ── factory methods ───────────────────────────────────────────────────────

    @classmethod
    def cold_start(cls, user_id: int, learning_rate: float = config.IRT_LEARNING_RATE) -> "UserSession":
        """Brand-new user — all skills start at DEFAULT_ABILITY."""
        return cls(user_id, IRTAbilityTracker(learning_rate=learning_rate))

    @classmethod
    def from_history(
        cls,
        user_id:       int,
        theta_path:    str   = config.THETA_FINAL_PATH,
        learning_rate: float = config.IRT_LEARNING_RATE,
    ) -> "UserSession":
        """
        Warm-start from theta_final.parquet (saved by the data generator).
        Falls back to cold start if the file or user is not found.
        """
        try:
            theta_df = pd.read_parquet(theta_path)
        except FileNotFoundError:
            return cls.cold_start(user_id, learning_rate)

        row = theta_df[theta_df["user_id"] == user_id]
        if row.empty:
            return cls.cold_start(user_id, learning_rate)

        skill_cols = [c for c in theta_df.columns if c.startswith("skill_")]
        initial    = {int(c.split("_")[1]): float(row.iloc[0][c]) for c in skill_cols}
        return cls(user_id, IRTAbilityTracker(initial_abilities=initial, learning_rate=learning_rate))

    # core loop 

    def next_task(self, engine, topic_filter: Optional[str] = None) -> Optional[Dict]:
        recs = engine.recommend(
            user_id         = self.user_id,
            ability_tracker = self.tracker,
            seen_task_ids   = self.seen,
            topic_filter    = topic_filter,
            n_return        = 1,
        )
        return recs[0] if recs else None

    def record_answer(self, task_id: int, task_meta: Dict, is_correct: bool) -> AnswerRecord:
        self._step   += 1
        primary       = int(task_meta.get("primary_skill", 0))
        secondary     = int(task_meta.get("secondary_skill", 0) or 0)
        b             = float(task_meta.get("difficulty_irt", config.DEFAULT_ABILITY))

        ability_before = self.tracker.effective_ability(primary, secondary)
        p_correct      = rasch_p_correct(ability_before, b)

        self.tracker.update(
            primary_skill      = primary,
            task_difficulty_irt= b,
            is_correct         = is_correct,
            secondary_skill    = secondary,
        )
        ability_after = self.tracker.effective_ability(primary, secondary)

        rec = AnswerRecord(
            step           = self._step,
            task_id        = task_id,
            topic_group    = str(task_meta.get("topic_group", "?")),
            primary_skill  = primary,
            difficulty_irt = round(b, 3),
            is_correct     = is_correct,
            ability_before = round(ability_before, 4),
            ability_after  = round(ability_after, 4),
            p_correct      = round(p_correct, 3),
            combined_score = round(float(task_meta.get("combined_score", 0.0)), 4),
            elapsed_sec    = round(time.time() - self._start, 1),
        )
        self.history.append(rec)
        self.seen.add(task_id)
        return rec

    # reporting

    def summary(self) -> Dict:
        if not self.history:
            return {"steps": 0}
        correct = sum(1 for r in self.history if r.is_correct)
        return {
            "user_id":          self.user_id,
            "steps":            self._step,
            "correct":          correct,
            "accuracy":         round(correct / self._step, 3),
            "avg_difficulty":   round(float(np.mean([r.difficulty_irt for r in self.history])), 3),
            "ability_start":    round(self.history[0].ability_before, 3),
            "ability_end":      round(self.history[-1].ability_after, 3),
            "ability_delta":    round(self.history[-1].ability_after - self.history[0].ability_before, 3),
            "skills_practised": len({r.primary_skill for r in self.history}),
            "duration_sec":     round(time.time() - self._start, 1),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "step":           r.step,
                "task_id":        r.task_id,
                "topic_group":    r.topic_group,
                "primary_skill":  r.primary_skill,
                "difficulty_irt": r.difficulty_irt,
                "is_correct":     r.is_correct,
                "ability_before": r.ability_before,
                "ability_after":  r.ability_after,
                "p_correct":      r.p_correct,
                "combined_score": r.combined_score,
                "elapsed_sec":    r.elapsed_sec,
            }
            for r in self.history
        ])
