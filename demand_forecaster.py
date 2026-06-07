import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model():
    model = Sequential()
    model.add(
        LSTM(
            50,
            activation="relu",
            input_shape=(10,1)
        )
    )
    model.add(Dense(1))
    model.compile(
        optimizer="adam",
        loss="mse"
    )
    return model

historical_demand = np.array([
    100,120,140,130,150,
    170,180,160,190,210
])
X = []
y = []
for i in range(len(historical_demand)-10):
    X.append(
        historical_demand[i:i+10]
    )
    y.append(
        historical_demand[i+10]
    )
X = np.array(X)
y = np.array(y)
X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)
model = build_model()
if len(X) > 0:
    model.fit(
        X,
        y,
        epochs=20,
        verbose=1
    )
    model.save(
        "demand_forecaster.h5"
    )