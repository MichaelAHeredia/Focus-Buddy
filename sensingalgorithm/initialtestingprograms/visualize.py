import matplotlib.pyplot as plt
from config import MODEL

def plot_predictions(df, X_pred, pred_temp, pred_sound):
    """Plot predicted vs actual temperature and sound."""
    plt.figure(figsize=(10, 5))
    plt.plot(X_pred["time"], pred_temp, label="Predicted optimal temp")
    plt.plot(X_pred["time"], pred_sound, label="Predicted optimal sound")
    # plot for each day of the actual temp sound and focus, but all on the same "day" x-axis
    for i in range(df["day"].nunique()):
        day_df = df[df["day"] == i]
        plt.plot(day_df["time"] + i * (df["time"].max() + 1), day_df["temp"], "--", label=f"Actual temp Day {i}")
        plt.plot(day_df["time"] + i * (df["time"].max() + 1), day_df["sound"], "--", label=f"Actual sound Day {i}")
        plt.plot(day_df["time"] + i * (df["time"].max() + 1), day_df["focus"], ".", label=f"Focus level Day {i}")
    # plt.plot(df["time"], df["temp"], "--", label="Actual temp")
    # plt.plot(df["time"], df["sound"], "--", label="Actual sound")
    # plt.plot(df["time"], df["focus"], ".", label="Focus level")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Predicted Optimal Conditions vs Time Using %s Model" % MODEL.capitalize())
    plt.show()
