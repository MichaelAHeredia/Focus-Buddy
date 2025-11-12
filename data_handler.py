import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import SAMPLES_PER_DAY, FOCUS_VALUE, ROLL_WINDOW


def generate_synthetic_data(n_days: int = 1):
    """Generate synthetic example dataset for one or more days.

    Returns a DataFrame with columns: 'focus', 'sound', 'temp', 'time', 'day'.
    - n_days: number of sequential days to generate (1..n)
    """
    n_points = int(SAMPLES_PER_DAY) * int(max(1, int(n_days)))
    # global time index and per-day time
    global_time = np.arange(n_points)
    day = global_time // SAMPLES_PER_DAY
    time_in_day = global_time % SAMPLES_PER_DAY

    # base signals per timestep (repeat per day), allow small random day offsets
    temp_base = 22 + 2 * np.sin(2 * np.pi * time_in_day / SAMPLES_PER_DAY)
    sound_base = 40 + 5 * np.cos(2 * np.pi * time_in_day / SAMPLES_PER_DAY)

    # add small day-specific noise so days aren't identical
    temp = temp_base + 0.2 * np.random.normal(size=n_points) + 0.1 * day
    sound = sound_base + 0.5 * np.random.normal(size=n_points) - 0.05 * day

    focus = (
        1
        - 0.02 * np.abs(temp - 22)
        - 0.01 * np.abs(sound - 42)
        + np.random.normal(0, 0.05, n_points)
    )
    focus = np.clip(focus, 0, 1)

    # normalize globally so models can learn across days
    normfocus = (focus - focus.min()) / (focus.max() - focus.min())
    normsound = (sound - sound.min()) / (sound.max() - sound.min())
    normtemp = (temp - temp.min()) / (temp.max() - temp.min())

    df = pd.DataFrame({
        "focus": normfocus,
        "sound": normsound,
        "temp": normtemp,
        "time": time_in_day,
        "day": day,
    })
    # convenience: global_time index
    df["global_time"] = global_time
    
    # visualize input data before training by plotting relevant features
    plt.figure(figsize=(10, 5))
    for i in range(n_days):
        day_df = df[df["day"] == i]
        plt.plot(day_df["time"] + i * SAMPLES_PER_DAY, day_df["temp"], "--", label=f"Day {i} temp")
        plt.plot(day_df["time"] + i * SAMPLES_PER_DAY, day_df["sound"], "--", label=f"Day {i} sound")
        plt.plot(day_df["time"] + i * SAMPLES_PER_DAY, day_df["focus"], ".", label=f"Day {i} focus")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Initial Conditions vs Time From Input Data")
    plt.show()
    
    return df


def prepare_features(df: pd.DataFrame):
    """Add engineered features.

    - Computes hour_sin/hour_cos using samples-per-day period.
    - Computes focus_roll per-day using ROLL_WINDOW.
    - If 'day' column is absent, assumes single-day data (day=0).
    """
    df = df.copy()
    if "day" not in df.columns:
        df["day"] = 0

    df["hour_sin"] = np.sin(2 * np.pi * df["time"] / SAMPLES_PER_DAY)
    df["hour_cos"] = np.cos(2 * np.pi * df["time"] / SAMPLES_PER_DAY)

    # compute rolling focus per day to avoid mixing days
    df["focus_roll"] = (
        df.groupby("day")["focus"]
        .rolling(ROLL_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # example day-level feature: day_of_week if day numbers represent calendar days
    df["day_of_week"] = df["day"] % 7
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def build_constant_focus_input(focus_value=FOCUS_VALUE, n_points=SAMPLES_PER_DAY, feature_columns=None):
    """Generate a synthetic input of constant focus (e.g., 0.9 all day).

    Parameters
    - focus_value: value for the 'focus' input (and 'focus_roll' by default)
    - n_points: number of time steps
    - feature_columns: optional list of column names to include. If provided,
      the function will try to populate known features (focus, focus_roll,
      hour_sin, hour_cos). Unknown feature names will be filled with zeros.
    """
    time_idx = np.arange(n_points)

    # default feature set
    # use SAMPLES_PER_DAY as the period so features align with training
    default = {
        "focus": np.full(n_points, focus_value),
        "focus_roll": np.full(n_points, focus_value),
        "hour_sin": np.sin(2 * np.pi * time_idx / SAMPLES_PER_DAY),
        "hour_cos": np.cos(2 * np.pi * time_idx / SAMPLES_PER_DAY),
    }

    if feature_columns is None:
        # return the default set (keeps previous behaviour)
        X_pred = pd.DataFrame(default)
        X_pred["time"] = time_idx
        X_pred["day"] = 0
        return X_pred

    # build dataframe containing requested columns
    data = {}
    for col in feature_columns:
        if col in default:
            data[col] = default[col]
        elif col == "time":
            data["time"] = time_idx
        else:
            # unknown feature: fill with zeros (caller can replace later)
            data[col] = np.zeros(n_points)

    # ensure focus and focus_roll exist if requested implicitly
    if "focus" in feature_columns and "focus_roll" not in data:
        data["focus_roll"] = np.full(n_points, focus_value)

    # if day requested but not provided, default to day 0
    if "day" in feature_columns and "day" not in data:
        data["day"] = np.zeros(n_points, dtype=int)

    X_pred = pd.DataFrame(data)
    # if time wasn't requested, still add it for plotting convenience
    if "time" not in X_pred.columns:
        X_pred["time"] = time_idx

    return X_pred
