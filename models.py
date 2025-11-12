from sklearn.linear_model import Ridge
import pandas as pd

def train_ridge_model(df):
    """Train ridge regression models for temp and sound."""
    X = df[["focus", "focus_roll", "hour_sin", "hour_cos"]]
    y_temp = df["temp"]
    y_sound = df["sound"]

    model_temp = Ridge().fit(X, y_temp)
    model_sound = Ridge().fit(X, y_sound)
    return model_temp, model_sound


def predict_conditions(model_temp, model_sound, X_pred):
    """Predict temperature and sound given focus input."""
    X_features = X_pred[["focus", "focus_roll", "hour_sin", "hour_cos"]]
    pred_temp = model_temp.predict(X_features)
    pred_sound = model_sound.predict(X_features)
    return pred_temp, pred_sound
