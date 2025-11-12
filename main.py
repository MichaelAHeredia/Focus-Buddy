from data_handler import generate_synthetic_data, prepare_features, build_constant_focus_input
from models import train_models, predict_from_models
from visualize import plot_predictions
from config import MODEL
import pandas as pd

def main():
    # TODO
    # implement algorithmic consideration on how long someone has been focused
        # for example, if someone's productivity will be increased in the future by taking breaks now
    # TODO
    # implement use of other ml models like decision trees or neural networks

    # Step 1: Prepare data
    df = generate_synthetic_data()
    df = prepare_features(df)

    # Step 2: Train models (choose any supported model_name)
    # model_name can be: 'ridge','linear','polynomial','random_forest','gbr','svr','xgboost'
    model_name = MODEL
    # determine feature columns automatically (all numeric except targets/time)
    target_columns = ["temp", "sound"]
    feature_columns = [
        c for c, dt in df.dtypes.items()
        if pd.api.types.is_numeric_dtype(dt) and c not in target_columns and c != "time"
    ]

    models = train_models(df, target_columns=target_columns, feature_columns=feature_columns, model_name=model_name)

    # Step 3: Generate constant-focus day input using the same feature columns
    X_pred = build_constant_focus_input(focus_value=0.9, n_points=len(df), feature_columns=feature_columns + ["time"])

    # Step 4: Predict conditions
    preds = predict_from_models(models, X_pred, feature_columns=feature_columns)
    pred_temp = preds["temp"]
    pred_sound = preds["sound"]

    # Step 5: Visualize results
    plot_predictions(df, X_pred, pred_temp, pred_sound)

if __name__ == "__main__":
    main()
