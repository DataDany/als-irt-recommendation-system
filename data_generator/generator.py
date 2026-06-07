"""
data_generator/generator.py
----------------------------
Generates all training data for the adaptive recommender system.

Each user has a Dirichlet-sampled topic-affinity vector over 8 topic groups (A–H).
Task selection blends:
  • 65 % topic-affinity-weighted sampling  → creates user clusters ALS can learn
  • 35 % skill-difficulty matching         → educational realism

Produces (in ../big_data/):
  interactions.parquet  — user × task interaction log
  tasks.parquet         — tasks with difficulty / skill / topic metadata
  users.parquet         — users with BKT params + topic affinities
  skills.parquet        — 100 Polish math skills
  theta_final.parquet   — per-user final skill abilities (for IRT warm-start)

Run:  python data_generator/generator.py
"""

import os, sys, time
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spark_utils import get_spark


# Config

@dataclass
class Config:
    seed: int = 42
    n_skills: int = 100
    n_tasks:  int = 5_000
    n_users:  int = 5_000

    interactions_per_user_mean: int = 400
    interactions_per_user_std:  int = 60

    prob_second_skill: float = 0.35
    prob_prereq:       float = 0.30
    max_prereq:        int   = 2

    target_p_low:  float = 0.35
    target_p_high: float = 0.70

    logistic_a:             float = 3.5
    secondary_skill_weight: float = 0.45

    guess_mean: float = 0.08;  guess_std: float = 0.03
    slip_mean:  float = 0.06;  slip_std:  float = 0.02
    learning_rate_mean: float = 0.035; learning_rate_std: float = 0.015

    base_time_seconds:   int = 35
    time_per_difficulty: int = 18

    topic_affinity_alpha: float = 0.5
    affinity_sample_fraction: float = 0.65
    sample_size: int = 20   # was 50 - smaller = much faster, still good signal


cfg = Config()
TOPICS = ["A", "B", "C", "D", "E", "F", "G", "H"]
N_TOPICS = len(TOPICS)


def sigmoid_vec(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def topic_group(skill_id: int) -> str:
    if skill_id <= 12: return "A"
    if skill_id <= 24: return "B"
    if skill_id <= 36: return "C"
    if skill_id <= 50: return "D"
    if skill_id <= 64: return "E"
    if skill_id <= 78: return "F"
    if skill_id <= 90: return "G"
    return "H"


def update_skill(theta: float, lr: float, is_correct: int) -> float:
    if is_correct:
        theta = theta + lr * (1.0 - theta)
    else:
        theta = theta + 0.25 * lr * (1.0 - theta)
    return float(np.clip(theta, 0.0, 1.0))


# Skills

SKILLS_BASE = [
    # A (1–12) — Arytmetyka
    "Kolejność działań", "Liczby całkowite – działania", "Ułamki zwykłe – działania",
    "Ułamki dziesiętne – działania", "Procenty – obliczenia podstawowe",
    "Procent składany / zmiana procentowa", "Potęgi o wykładniku całkowitym",
    "Pierwiastki – własności i przekształcenia", "Notacja naukowa i przybliżenia",
    "Wartość bezwzględna – podstawy", "Szacowanie i zaokrąglanie", "Jednostki i zamiana jednostek",
    # B (13–24) — Algebra
    "Redukcja wyrazów podobnych", "Wyłączanie wspólnego czynnika",
    "Mnożenie nawiasów", "Wzory skróconego mnożenia (a±b)²",
    "Wzory skróconego mnożenia (a-b)(a+b)", "Rozkład na czynniki – podstawy",
    "Ułamki algebraiczne – skracanie", "Ułamki algebraiczne – dod./odejm.",
    "Ułamki algebraiczne – mnoż./dziel.", "Równoważne przekształcenia wyrażeń",
    "Podstawianie do wzoru", "Przekształcanie wzorów",
    # C (25–36) — Równania
    "Równania liniowe 1 zmienna", "Równania liniowe z ułamkami",
    "Równania z wartością bezwzględną", "Układy równań (podstawianie)",
    "Układy równań (eliminacja)", "Nierówności liniowe",
    "Nierówności z ułamkami", "Nierówności z wartością bezwzględną",
    "Równania kwadratowe (ogólna)", "Równania kwadratowe (delta)",
    "Równania kwadratowe (rozkład)", "Nierówności kwadratowe",
    # D (37–50) — Funkcje
    "Pojęcie funkcji, dziedzina", "Odczyt z wykresu",
    "Odczyt z tabeli/diagramu", "Wykres funkcji liniowej y=ax+b",
    "Interpretacja współczynników a i b", "Równanie prostej przez 2 punkty",
    "Miejsce zerowe (liniowa)", "Funkcja kwadratowa – postacie i wykres",
    "Wierzchołek paraboli", "Miejsca zerowe paraboli",
    "Monotoniczność", "Przekształcenia wykresów",
    "Funkcja odwrotna", "Punkt przecięcia wykresów",
    # E (51–64) — Geometria płaska
    "Kąty w trójkącie", "Przystawanie trójkątów", "Podobieństwo trójkątów",
    "Twierdzenie Talesa", "Twierdzenie Pitagorasa", "Pitagoras w zadaniach złożonych",
    "Pole trójkąta", "Pole czworokątów", "Obwód i pole koła",
    "Kąty w okręgu", "Wielokąty foremne", "Symetrie",
    "Wnioskowanie na rysunku", "Równanie okręgu",
    # F (65–78) — Geometria przestrzenna
    "Graniastosłupy – pola i objętości", "Ostrosłupy – pola i objętości",
    "Walec – pole i objętość", "Stożek – pole i objętość", "Kula – pole i objętość",
    "Przekroje brył", "Siatki brył", "Pitagoras w bryłach",
    "Skala podobieństwa brył", "Zadania tekstowe z brył",
    "Geometria analityczna: odległość", "Geometria analityczna: środek odcinka",
    "Modelowanie zależności liniowych", "Współrzędne: odległość (powt.)",
    # G (79–90) — Trygonometria
    "Sin/cos/tan w trójkącie prostokątnym", "Zależności między sin/cos/tan",
    "Obliczanie długości/kątów", "Pole trójkąta z sinusem",
    "Kąt nachylenia", "Proste zadania trygonometryczne",
    "Trygonometria w geometrii płaskiej", "Trygonometria w geometrii przestrzennej",
    "Funkcje trygonometryczne – wykresy", "Tożsamości trygonometryczne",
    "Równania trygonometryczne", "Zastosowania trygonometrii",
    # H (91–100) — Statystyka
    "Średnia arytmetyczna i ważona", "Mediana, dominanta",
    "Rozstęp i intuicja zmienności", "Interpretacja wykresów/diagramów",
    "Kombinatoryka: zasada mnożenia/dodawania", "Permutacje i kombinacje",
    "Prawdopodobieństwo klasyczne", "Prawdopodobieństwo warunkowe",
    "Zdarzenia niezależne", "Rozkład dwumianowy (podstawy)",
]

skills = SKILLS_BASE[:cfg.n_skills]
if len(skills) < cfg.n_skills:
    skills += [f"skill_{i}" for i in range(len(skills) + 1, cfg.n_skills + 1)]

skills_pdf = pd.DataFrame({
    "skill_id":   np.arange(1, len(skills) + 1, dtype=int),
    "skill_name": skills,
})


# Tasks 

rng = np.random.default_rng(cfg.seed)

task_id       = np.arange(1, cfg.n_tasks + 1, dtype=int)
primary_skill = rng.integers(1, cfg.n_skills + 1, size=cfg.n_tasks)
has_secondary = rng.random(cfg.n_tasks) < cfg.prob_second_skill
secondary_skill = np.where(
    has_secondary, rng.integers(1, cfg.n_skills + 1, size=cfg.n_tasks), 0
).astype(int)

base_diff = np.clip(rng.normal(5.0, 2.0, cfg.n_tasks), 1.0, 10.0)
group     = np.array([topic_group(int(s)) for s in primary_skill])
diff      = base_diff.copy()
diff[group == "A"] -= 0.4;  diff[group == "H"] -= 0.2
diff[group == "C"] += 0.3;  diff[group == "D"] += 0.2
diff = np.clip(diff, 1.0, 10.0)

prereq_1 = np.zeros(cfg.n_tasks, dtype=int)
prereq_2 = np.zeros(cfg.n_tasks, dtype=int)
for i in range(cfg.n_tasks):
    if rng.random() >= cfg.prob_prereq: continue
    s1 = int(primary_skill[i])
    cands = np.arange(1, s1) if s1 > 1 else np.array([], dtype=int)
    if len(cands) == 0: continue
    chosen = rng.choice(cands, size=min(int(rng.integers(1, cfg.max_prereq + 1)), len(cands)), replace=False)
    prereq_1[i] = int(chosen[0])
    if len(chosen) > 1: prereq_2[i] = int(chosen[1])

tasks_pdf = pd.DataFrame({
    "task_id":        task_id,
    "topic_group":    group,
    "primary_skill":  primary_skill.astype(int),
    "secondary_skill": secondary_skill,
    "difficulty":     diff.astype(float),
    "prereq_skill_1": prereq_1,
    "prereq_skill_2": prereq_2,
})

# Precompute numpy arrays for vectorized task selection
_tasks_s1   = tasks_pdf["primary_skill"].to_numpy(dtype=np.int32)
_tasks_s2   = tasks_pdf["secondary_skill"].to_numpy(dtype=np.int32)
_tasks_diff = tasks_pdf["difficulty"].to_numpy(dtype=np.float32)
_tasks_tg   = tasks_pdf["topic_group"].to_numpy()
_tasks_id   = tasks_pdf["task_id"].to_numpy(dtype=np.int32)

# topic - list of 0-based row indices
topic_to_task_indices: dict = {
    t: np.where(_tasks_tg == t)[0]
    for t in TOPICS
}


# Users

rng_u   = np.random.default_rng(cfg.seed + 1)
user_id = np.arange(1, cfg.n_users + 1, dtype=int)
guess   = np.clip(rng_u.normal(cfg.guess_mean, cfg.guess_std, cfg.n_users), 0.01, 0.25)
slip    = np.clip(rng_u.normal(cfg.slip_mean,  cfg.slip_std,  cfg.n_users), 0.01, 0.25)
learning_rate = np.clip(
    rng_u.normal(cfg.learning_rate_mean, cfg.learning_rate_std, cfg.n_users), 0.005, 0.12
)

topic_affinity = rng_u.dirichlet(
    alpha=np.full(N_TOPICS, cfg.topic_affinity_alpha),
    size=cfg.n_users,
)  # shape (n_users, 8)

topic_aff_cols = {f"topic_aff_{t}": topic_affinity[:, i] for i, t in enumerate(TOPICS)}

users_pdf = pd.DataFrame({
    "user_id":       user_id,
    "guess":         guess.astype(float),
    "slip":          slip.astype(float),
    "learning_rate": learning_rate.astype(float),
    **topic_aff_cols,
})

theta_init = np.clip(
    rng_u.beta(2.0, 6.0, (cfg.n_users, cfg.n_skills)), 0.0, 1.0
).astype(np.float32)


# Vectorized task selection

def choose_task(
    theta_row: np.ndarray,
    user_topic_aff: np.ndarray,
    user_guess: float,
    user_slip: float,
    rng_local: np.random.Generator,
) -> tuple:
    """
    Vectorized task selection — no iterrows, no pandas access in inner loop.
    Returns (task_id, p_correct).
    """
    n = cfg.sample_size
    n_aff = int(round(n * cfg.affinity_sample_fraction))
    n_rnd = n - n_aff

    idx = []
    if n_aff > 0:
        topic_draws = rng_local.choice(N_TOPICS, size=n_aff, p=user_topic_aff)
        for ti in topic_draws:
            pool = topic_to_task_indices[TOPICS[ti]]
            if len(pool) > 0:
                idx.append(int(pool[rng_local.integers(0, len(pool))]))
            else:
                idx.append(int(rng_local.integers(0, cfg.n_tasks)))
    if n_rnd > 0:
        idx.extend(rng_local.integers(0, cfg.n_tasks, size=n_rnd).tolist())

    idx_arr = np.array(idx, dtype=np.int32)

    # IRT scoring
    s1  = _tasks_s1[idx_arr]
    s2  = _tasks_s2[idx_arr]
    d   = (_tasks_diff[idx_arr] - 1.0) / 9.0

    level = theta_row[s1 - 1].astype(np.float64)
    mask  = s2 != 0
    level[mask] = (
        (1 - cfg.secondary_skill_weight) * level[mask]
        + cfg.secondary_skill_weight * theta_row[s2[mask] - 1].astype(np.float64)
    )

    base  = sigmoid_vec(cfg.logistic_a * (level - d))
    p_arr = np.clip(user_guess + (1.0 - user_guess - user_slip) * base, 0.0, 1.0)

    target = (cfg.target_p_low + cfg.target_p_high) / 2.0
    j = int(rng_local.integers(0, n)) if rng_local.random() < 0.08 else int(np.argmin(np.abs(p_arr - target)))

    return int(_tasks_id[idx_arr[j]]), float(p_arr[j])


# simulation

print(f"\nSimulating {cfg.n_users} users × ~{cfg.interactions_per_user_mean} interactions …")
t0 = time.time()

all_interactions = []
theta_final_rows = []

for i, uid in enumerate(range(1, cfg.n_users + 1)):
    if i % 500 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (cfg.n_users - i)
        print(f"  {i:>5}/{cfg.n_users}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", end="\r")

    ug = float(users_pdf.at[i, "guess"])
    us = float(users_pdf.at[i, "slip"])
    lr = float(users_pdf.at[i, "learning_rate"])
    user_topic_aff = topic_affinity[i].copy()
    user_topic_aff /= user_topic_aff.sum()

    theta_row = theta_init[i].copy().astype(np.float64)

    rng_local = np.random.default_rng(cfg.seed + uid)
    n_int = int(np.clip(
        rng_local.normal(cfg.interactions_per_user_mean, cfg.interactions_per_user_std),
        5, cfg.interactions_per_user_mean + 4 * cfg.interactions_per_user_std,
    ))

    t = datetime(2025, 1, 1, 8, 0, 0) + timedelta(seconds=int(rng_local.integers(0, 3600)))

    for _ in range(n_int):
        tid, op = choose_task(theta_row, user_topic_aff, ug, us, rng_local)
        row_idx  = tid - 1
        s1 = int(_tasks_s1[row_idx]);  s2 = int(_tasks_s2[row_idx])
        d  = float(_tasks_diff[row_idx]); tg = str(_tasks_tg[row_idx])

        correct  = int(rng_local.random() < op)
        attempts = 1 + int(rng_local.random() < (1-op)*0.6) + int(rng_local.random() < (1-op)*0.25)
        hints    = int(rng_local.random() < (1-op)*0.35) + int(rng_local.random() < (1-op)*0.15)
        ts       = float(max(20.0,
                    cfg.base_time_seconds + cfg.time_per_difficulty * d * (1 + 0.25*(1-op))
                    + rng_local.normal(0, 12)))

        theta_row[s1-1] = update_skill(float(theta_row[s1-1]), lr, correct)
        if s2 != 0:
            theta_row[s2-1] = update_skill(float(theta_row[s2-1]), lr * 0.6, correct)

        t += timedelta(seconds=int(max(20, ts)) + int(rng_local.integers(5, 120)))
        all_interactions.append((uid, tid, t.isoformat(), correct, attempts, ts, hints, op,
                                  s1, s2, d, tg))

    th = {"user_id": uid}
    for k in range(cfg.n_skills):
        th[f"skill_{k+1}"] = float(theta_row[k])
    theta_final_rows.append(th)

elapsed_total = time.time() - t0
print(f"\n  Done — {len(all_interactions):,} interactions in {elapsed_total:.1f}s")

# Build DataFrames

interactions_pdf = pd.DataFrame(all_interactions, columns=[
    "user_id", "task_id", "timestamp", "is_correct", "attempts",
    "time_spent_sec", "hints_used", "oracle_p_correct",
    "primary_skill", "secondary_skill", "difficulty", "topic_group",
])

theta_final_pdf = pd.DataFrame(theta_final_rows)


# Write to parquet via Spark (keeps format compatible with ALS pipeline) 

print("\nWriting parquet files …")
spark = get_spark("generate_data", driver_memory="4g")
spark.sparkContext.setLogLevel("WARN")

out_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "big_data"))
os.makedirs(out_base, exist_ok=True)

spark.createDataFrame(skills_pdf).write.mode("overwrite").parquet(f"{out_base}/skills.parquet")
spark.createDataFrame(tasks_pdf).write.mode("overwrite").parquet(f"{out_base}/tasks.parquet")
spark.createDataFrame(users_pdf).write.mode("overwrite").parquet(f"{out_base}/users.parquet")
spark.createDataFrame(interactions_pdf).write.mode("overwrite").parquet(f"{out_base}/interactions.parquet")
spark.createDataFrame(theta_final_pdf).write.mode("overwrite").parquet(f"{out_base}/theta_final.parquet")

print(f"\nDone. Data written to {out_base}/")
print(f"  tasks={cfg.n_tasks:,}  users={cfg.n_users:,}  "
      f"interactions={len(all_interactions):,}  "
      f"density={len(all_interactions)/(cfg.n_tasks*cfg.n_users)*100:.1f}%")
print(f"Topic affinity α={cfg.topic_affinity_alpha}  "
      f"(lower = more peaked user preferences)")

spark.stop()

# Write CSV copies

print("\nWriting CSV files …")
csv_base = os.path.join(out_base, "csv")
os.makedirs(csv_base, exist_ok=True)

skills_pdf.to_csv(f"{csv_base}/skills.csv", index=False)
tasks_pdf.to_csv(f"{csv_base}/tasks.csv", index=False)
users_pdf.to_csv(f"{csv_base}/users.csv", index=False)
interactions_pdf.to_csv(f"{csv_base}/interactions.csv", index=False)
theta_final_pdf.to_csv(f"{csv_base}/theta_final.csv", index=False)

print(f"CSV files written to {csv_base}/")
print(f"  skills.csv       {len(skills_pdf):>7,} rows")
print(f"  tasks.csv        {len(tasks_pdf):>7,} rows")
print(f"  users.csv        {len(users_pdf):>7,} rows")
print(f"  interactions.csv {len(interactions_pdf):>7,} rows")
print(f"  theta_final.csv  {len(theta_final_pdf):>7,} rows")
