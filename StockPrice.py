import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

data = np.sin(np.arange(1000) * 0.01)  # Sine wave as dummy data
X = []
y = []
for i in range(len(data) - 10):
    X.append(data[i:i + 10])
    y.append(data[i + 10])

X = np.array(X).reshape(-1, 10, 1)
y = np.array(y)

model = Sequential()
model.add(SimpleRNN(50, activation='tanh', input_shape=(10, 1)))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=10, verbose=1)

predictions = model.predict(X)
print(predictions)
