# Configuration file

DATA_PATH = "data/focus_day.csv"  # placeholder for future file input
MODEL = "ridge"  # default model type
# Sampling parameters
SAMPLES_PER_DAY = 100   # 24 hours * 12 (5-min intervals)
FOCUS_VALUE = 0.9
ROLL_WINDOW = 3
# Number of days to generate/train on by default for multi-day experiments
N_DAYS = 3
