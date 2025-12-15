import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# TODO
# implement algorithmic consideration on how long someone has been focused
    # for example, if someone's productivity will e increased in the future by taking breaks now
    
# implement use of other ml models like decision trees or neural networks


# using ridge regression this algorithm will generate 
# a predictive model for the optimal conditions for productivity

# generate some input data
focus = np.random.randint(0,2,100)
sound = np.random.randint(25,56,100)
temp = np.random.randint(60,81,100)
timestamps = np.linspace(0,24,100)

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

df['hour_sin'] = np.sin(2 * np.pi * df['time'] / (24*12))
df['hour_cos'] = np.cos(2 * np.pi * df['time'] / (24*12))
df['focus_roll'] = df['focus'].rolling(3, min_periods=1).mean()

# plot the input data
# plt.scatter(df['focus'],df['time'], label="Focus")
# plt.scatter(df['focus_roll'],df['time'], label="Rolling Focus average")
# plt.scatter(df['sound'],df['time'], label="Sound")
# plt.scatter(df['temp'],df['time'], label= "Temp")
# plt.xlabel('Inputs and Focus')
# plt.ylabel('Time')
# plt.title('Input Data over Time')
# plt.legend()
# plt.show()

# use ridge regression
X = df[['focus', 'focus_roll', 'hour_sin', 'hour_cos']]
y_temp = df['temp']
y_sound = df['sound']

model_temp = Ridge().fit(X, y_temp)
model_sound = Ridge().fit(X, y_sound)

focus_value = 0.9
samples_per_day = len(df)
time_idx = np.arange(samples_per_day)

# place in predictive dataframe
X_pred = pd.DataFrame({
    'focus': np.full(samples_per_day, focus_value),
    'focus_roll': np.full(samples_per_day, focus_value),
    'hour_sin': np.sin(2 * np.pi * time_idx / (samples_per_day)),
    'hour_cos': np.cos(2 * np.pi * time_idx / (samples_per_day))
})

# Predict optimal conditions given focus sequence
pred_temp = model_temp.predict(X_pred)
pred_sound = model_sound.predict(X_pred)
print(pred_temp.shape)
print(pred_sound.shape)

plt.plot(df['time'], pred_temp, label="Predicted optimal temperature")
plt.plot(df['time'], pred_sound, label="Predicted optimal noise level")
plt.plot(df['time'], df['temp'], '--', label="Actual temperature")
plt.plot(df['time'], df['sound'], '--', label="Actual noise level")
plt.plot(df['time'], df['focus'], '.', label="Focus level")
plt.xlabel('Time')
plt.ylabel('Temperature / Noise Level')
plt.legend()
plt.title('Predicted Optimal Conditions vs Time')
plt.show()

