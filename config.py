# Configuration file

DATA_PATH = "data/focus_day.csv"  # placeholder for future file input
MODEL_TYPE = "ridge"  # could later be "xgboost", "lstm", etc.

# Sampling parameters
SAMPLES_PER_DAY = 100   # 24 hours * 12 (5-min intervals)
FOCUS_VALUE = 0.9
ROLL_WINDOW = 3
