import pandas as pd
import numpy as np
from config import SAMPLES_PER_DAY, FOCUS_VALUE, ROLL_WINDOW

def generate_synthetic_data():
    """Generate a synthetic example dataset."""
    # generate some input data
    time_idx = np.arange(SAMPLES_PER_DAY)
    focus = np.random.uniform(0,2,SAMPLES_PER_DAY)
    temp = 22 + 2 * np.sin(2 * np.pi * time_idx / SAMPLES_PER_DAY)
    sound = 40 + 5 * np.cos(2 * np.pi * time_idx / SAMPLES_PER_DAY)

    # normalize the data
    normfocus = (focus - focus.min()) / (focus.max() - focus.min())
    normsound = (sound - sound.min()) / (sound.max() - sound.min())
    normtemp = (temp - temp.min()) / (temp.max() - temp.min())

    # start by refining input data
    df = pd.DataFrame({
        'focus': normfocus,
        'sound': normsound,
        'temp': normtemp,
        'time': np.arange(len(focus))
    })
    return df


def prepare_features(df):
    """Add engineered features."""
    df["hour_sin"] = np.sin(2 * np.pi * df["time"] / SAMPLES_PER_DAY)
    df["hour_cos"] = np.cos(2 * np.pi * df["time"] / SAMPLES_PER_DAY)
    df["focus_roll"] = df["focus"].rolling(ROLL_WINDOW, min_periods=1).mean()
    return df


def build_constant_focus_input(focus_value=FOCUS_VALUE, n_points=SAMPLES_PER_DAY):
    """Generate a synthetic input of constant focus (e.g., 0.9 all day)."""
    time_idx = np.arange(n_points)
    X_pred = pd.DataFrame({
        "focus": np.full(n_points, focus_value),
        "focus_roll": np.full(n_points, focus_value),
        "hour_sin": np.sin(2 * np.pi * time_idx / n_points),
        "hour_cos": np.cos(2 * np.pi * time_idx / n_points)
    })
    X_pred["time"] = time_idx
    return X_pred
