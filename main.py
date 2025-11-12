from data_handler import generate_synthetic_data, prepare_features, build_constant_focus_input
from models import train_ridge_model, predict_conditions
from visualize import plot_predictions

def main():
    # Step 1: Prepare data
    df = generate_synthetic_data()
    df = prepare_features(df)

    # Step 2: Train model
    model_temp, model_sound = train_ridge_model(df)

    # Step 3: Generate constant-focus day input
    X_pred = build_constant_focus_input(focus_value=0.9)

    # Step 4: Predict conditions
    pred_temp, pred_sound = predict_conditions(model_temp, model_sound, X_pred)

    # Step 5: Visualize results
    plot_predictions(df, X_pred, pred_temp, pred_sound)

if __name__ == "__main__":
    main()
