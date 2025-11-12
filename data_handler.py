import pandas as pd
import numpy as np
from config import SAMPLES_PER_DAY, FOCUS_VALUE, ROLL_WINDOW

def generate_synthetic_data():
    """Generate a synthetic example dataset."""
    # generate some input data
    time_idx = np.arange(SAMPLES_PER_DAY)
    temp = 22 + 2 * np.sin(2*np.pi*time_idx/SAMPLES_PER_DAY)
    sound = 40 + 5 * np.cos(2*np.pi*time_idx/SAMPLES_PER_DAY)
    focus = 1 - 0.02*np.abs(temp - 22) - 0.01*np.abs(sound - 42) + np.random.normal(0, 0.05, SAMPLES_PER_DAY)
    focus = np.clip(focus, 0, 1)
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
    default = {
        "focus": np.full(n_points, focus_value),
        "focus_roll": np.full(n_points, focus_value),
        "hour_sin": np.sin(2 * np.pi * time_idx / n_points),
        "hour_cos": np.cos(2 * np.pi * time_idx / n_points),
    }

    if feature_columns is None:
        # return the default set (keeps previous behaviour)
        X_pred = pd.DataFrame(default)
        X_pred["time"] = time_idx
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

    X_pred = pd.DataFrame(data)
    # if time wasn't requested, still add it for plotting convenience
    if "time" not in X_pred.columns:
        X_pred["time"] = time_idx

    return X_pred
