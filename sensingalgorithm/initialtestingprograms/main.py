from data_handler import generate_synthetic_data, prepare_features, build_constant_focus_input
from models import train_models, predict_from_models
from sensingalgorithm.initialtestingprograms.visualize import plot_predictions
from config import MODEL, N_DAYS, SAMPLES_PER_DAY
import pandas as pd

def main():
    # TODO
    # implement algorithmic consideration on how long someone has been focused
        # for example, if someone's productivity will be increased in the future by taking breaks now
    # TODO
    # implement use of other ml models like decision trees or neural networks

    # Prepare multi-day training data (set N_DAYS in config.py or change here)
    df = generate_synthetic_data(n_days=N_DAYS)
    df = prepare_features(df)

    # Train models (choose any supported model_name)
    # model_name can be: 'ridge','linear','polynomial','random_forest','gbr','svr','xgboost'
    model_name = MODEL
    
    # determine feature columns automatically (all numeric except targets/time)
    target_columns = ["temp", "sound"]
    feature_columns = [
        c for c, dt in df.dtypes.items()
        if pd.api.types.is_numeric_dtype(dt) and c not in target_columns and c != "time"
    ]

    models = train_models(df, target_columns=target_columns, feature_columns=feature_columns, model_name=model_name)

    # Generate constant-focus single-day input using the same feature columns
    # Prediction should be produced for one day's worth of timesteps
    X_pred = build_constant_focus_input(focus_value=0.9, n_points=SAMPLES_PER_DAY, feature_columns=feature_columns + ["time", "day"])

    # Predict conditions
    preds = predict_from_models(models, X_pred, feature_columns=feature_columns)
    pred_temp = preds["temp"]
    pred_sound = preds["sound"]

    # Visualize results
    plot_predictions(df, X_pred, pred_temp, pred_sound)

if __name__ == "__main__":
    main()
