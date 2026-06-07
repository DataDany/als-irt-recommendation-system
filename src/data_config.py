# ALS training data 
DATA_PATH         = "big_data/interactions.parquet/"
QUALITY_DATA_PATH = "big_data/als_quality_ratings.parquet"
OUT_DIR           = "big_data/results"

# Auxiliary tables
TASKS_PATH        = "big_data/tasks.parquet"
THETA_FINAL_PATH  = "big_data/theta_final.parquet"

# IRT calibration output
IRT_PARAMS_PATH   = "big_data/irt_params.parquet"

# ALS factor export (numpy) — fast inference without Spark
USER_FACTORS_PATH = "big_data/als_user_factors.npz"
ITEM_FACTORS_PATH = "big_data/als_item_factors.npz"

# ALS column names 
SEED       = 42
TOP_K      = 10
USER_COL   = "user_id"
ITEM_COL   = "task_id"
RATING_COL = "rating"

# "binary"        → each interaction = 1.0 (pure implicit count)
# "correct_count" → counts only correct solves; multiple correct = higher weight
# "log_correct"   → log(1 + correct_count) — dampens power users, cleanest signal
# "quality"       → correctness weighted by attempts / hints / time
RATING_MODE = "log_correct"

# Evaluation
EVAL_K_LIST = [10, 20, 50]   # report metrics at all three cutoffs
TOP_K       = 10             # used as the primary K throughout the codebase

ALS_PARAMS = {
    "rank":          50,    # tuned via 70/15/15 grid search (tune_als.py --quick)
    "regParam":      0.05,  # tuned
    "maxIter":       20,
    "implicitPrefs": True,
    "nonnegative":   True,
    "alpha":         10.0,  # tuned
}

# Adaptive engine weights
# Final score = ALS_WEIGHT × als_score  +  IRT_WEIGHT × irt_difficulty_match
ALS_WEIGHT = 0.6
IRT_WEIGHT = 0.4

# IRT / session parameters
IRT_LEARNING_RATE     = 0.05   # θ step size after each answer
DEFAULT_ABILITY       = 0.30   # cold-start θ ∈ [0,1]
SECONDARY_SKILL_WEIGHT = 0.45  # weight of secondary skill when blending θ
N_CANDIDATES          = 50     # ALS pre-selection before IRT reranking
