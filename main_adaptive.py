"""
main_adaptive.py
----------------
Step 3 — Run an adaptive session.

Modes
-----
  --demo                        Build tiny synthetic data and simulate a session
  --user N [--topic X] [--steps N] [--simulate]
                                Session for a specific user (uses real data if available)
  --calibrate                   Re-run IRT calibration (delegates to calibrate_irt.py)

Examples
--------
  python main_adaptive.py --demo
  python main_adaptive.py --user 42 --topic C --steps 15 --simulate
  python main_adaptive.py --user 42 --topic C --steps 10      # interactive
  python main_adaptive.py --calibrate
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import subprocess

import numpy as np
import pandas as pd

import src.data_config as config
from src.irt_model import IRTAbilityTracker, rasch_p_correct
from src.session import UserSession

TOPIC_NAMES = {
    "A": "Arytmetyka i liczby",
    "B": "Wyrażenia algebraiczne",
    "C": "Równania i nierówności",
    "D": "Funkcje i geometria analityczna",
    "E": "Geometria płaska",
    "F": "Geometria przestrzenna",
    "G": "Trygonometria",
    "H": "Statystyka i prawdopodobieństwo",
}


# display helpers

def _bar(v: float, w: int = 20) -> str:
    f = round(float(v) * w)
    return "[" + "█" * f + "░" * (w - f) + f"] {float(v):.2f}"


def _header(title: str) -> None:
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)


# synthetic data builder (demo without Kaggle dataset

def _remove_if_dir(path: str) -> None:
    """Remove path if it is a Spark-created parquet directory."""
    if os.path.isdir(path):
        shutil.rmtree(path)


def _build_synthetic(n_tasks=500, n_users=200, rank=20, seed=config.SEED):
    os.makedirs("big_data", exist_ok=True)
    rng = np.random.default_rng(seed)

    # ALS factors
    user_ids     = np.arange(1, n_users + 1, dtype=np.int32)
    item_ids     = np.arange(1, n_tasks + 1, dtype=np.int32)
    user_factors = rng.random((n_users, rank)).astype(np.float32)
    item_factors = rng.random((n_tasks, rank)).astype(np.float32)
    np.savez_compressed(config.USER_FACTORS_PATH, ids=user_ids, factors=user_factors)
    np.savez_compressed(config.ITEM_FACTORS_PATH, ids=item_ids, factors=item_factors)

    # IRT params
    topics         = list(TOPIC_NAMES.keys())
    n_skills       = 100
    primary_skill  = rng.integers(1, n_skills + 1, size=n_tasks)
    sec_mask       = rng.random(n_tasks) < 0.35
    secondary_skill= np.where(sec_mask, rng.integers(1, n_skills + 1, size=n_tasks), 0)
    topic_group    = np.array([topics[int(s - 1) // 13] for s in primary_skill])
    diff_irt       = rng.beta(2, 2, size=n_tasks).clip(0.05, 0.95)

    irt_pd = pd.DataFrame({
        "task_id":           item_ids,
        "primary_skill":     primary_skill.astype(int),
        "secondary_skill":   secondary_skill.astype(int),
        "topic_group":       topic_group,
        "difficulty_raw":    (diff_irt * 9 + 1).round(2),
        "difficulty_norm":   diff_irt.round(4),
        "p_correct_empirical": (1 - diff_irt).round(4),
        "n_interactions":    rng.integers(50, 500, size=n_tasks),
        "difficulty_irt":    diff_irt.round(4),
    })
    _remove_if_dir(config.IRT_PARAMS_PATH)
    irt_pd.to_parquet(config.IRT_PARAMS_PATH, index=False)

    # theta_final (warm-start abilities)
    skill_cols = {f"skill_{i}": rng.beta(2, 5, size=n_users).clip(0.01, 0.99)
                  for i in range(1, n_skills + 1)}
    theta_pd = pd.DataFrame({"user_id": user_ids, **skill_cols})
    _remove_if_dir(config.THETA_FINAL_PATH)
    theta_pd.to_parquet(config.THETA_FINAL_PATH, index=False)

    print(f"  Synthetic data: {n_users} users, {n_tasks} tasks, rank={rank}")


def _ensure_data():
    required = [config.USER_FACTORS_PATH, config.ITEM_FACTORS_PATH, config.IRT_PARAMS_PATH]
    if all(os.path.exists(p) for p in required):
        return True
    print("[Info] Required files not found — generating synthetic data …")
    _build_synthetic()
    return False


# session runner

def run_session(user_id: int, topic: str | None, n_steps: int, auto_sim: bool, real_data: bool):
    from src.adaptive_engine import AdaptiveEngine

    _header(f"Adaptive Session  —  user {user_id}")

    engine = AdaptiveEngine()

    if real_data and os.path.exists(config.THETA_FINAL_PATH):
        session = UserSession.from_history(user_id)
    else:
        session = UserSession.cold_start(user_id)

    if topic:
        print(f"  Topic: {topic} — {TOPIC_NAMES.get(topic, topic)}")

    for step in range(1, n_steps + 1):
        rec = session.next_task(engine, topic_filter=topic)
        if rec is None:
            print("No more suitable tasks. Session ended early.")
            break

        print(f"\n── Step {step:>2}/{n_steps} " + "─" * 44)
        print(f"  Task #{rec['task_id']:>6}  Topic: {rec['topic_group']}  Skill: {rec['primary_skill']}")
        print(f"  Difficulty (IRT) : {_bar(rec['difficulty_irt'])}")

        theta = session.tracker.effective_ability(rec["primary_skill"], rec["secondary_skill"])
        p     = rasch_p_correct(theta, rec["difficulty_irt"])
        print(f"  Your ability     : {_bar(theta)}")
        print(f"  P(correct)       : {p:.1%}")
        print(f"  Score  ALS={rec['als_score']:.4f}  IRT={rec['irt_score']:.4f}  "
              f"Combined={rec['combined_score']:.4f}")

        if auto_sim:
            is_correct = bool(np.random.random() < p)
            print(f"\n  [AUTO] {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        else:
            ans = input("\n  Correct? [y/n]: ").strip().lower()
            is_correct = ans in ("y", "yes", "1")

        record = session.record_answer(rec["task_id"], rec, is_correct)
        delta  = record.ability_after - record.ability_before
        sign   = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  Ability (skill {rec['primary_skill']}): "
              f"{record.ability_before:.3f} {sign} {record.ability_after:.3f}  "
              f"(Δ {delta:+.4f})")

    s = session.summary()
    _header("Session Summary")
    print(f"  Steps         : {s['steps']}")
    print(f"  Accuracy      : {s['accuracy']:.1%}  ({s['correct']}/{s['steps']} correct)")
    print(f"  Avg difficulty: {_bar(s['avg_difficulty'])}")
    print(f"  Ability change: {s['ability_start']:.3f} → {s['ability_end']:.3f}  "
          f"(Δ {s['ability_delta']:+.3f})")
    print(f"  Skills seen   : {s['skills_practised']}")
    print(f"  Duration      : {s['duration_sec']:.0f}s")

    hist_path = f"big_data/results/session_{user_id}_history.parquet"
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    session.to_dataframe().to_parquet(hist_path, index=False)
    print(f"\n  History → {hist_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate", action="store_true",
                   help="Run offline IRT calibration (needs Spark + big_data/).")
    p.add_argument("--demo",      action="store_true",
                   help="Fully synthetic demo — no Kaggle data needed.")
    p.add_argument("--user",      type=int,  default=1)
    p.add_argument("--topic",     type=str,  default=None,
                   help="Topic filter A–H (default: all).")
    p.add_argument("--steps",     type=int,  default=10)
    p.add_argument("--simulate",  action="store_true",
                   help="Auto-simulate answers (no keyboard input).")
    args = p.parse_args()

    if args.calibrate:
        result = subprocess.run(
            [sys.executable, "data_generator/calibrate_irt.py"], check=False
        )
        sys.exit(result.returncode)

    if args.demo:
        _header("Demo Mode — synthetic data")
        _build_synthetic()
        real_data = False
    else:
        real_data = _ensure_data()

    run_session(
        user_id  = args.user,
        topic    = args.topic,
        n_steps  = args.steps,
        auto_sim = args.simulate or args.demo,
        real_data= real_data,
    )


if __name__ == "__main__":
    main()
