import numpy as np
import tensorflow as tf 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse=False)
y = y.reshape(-1, 1)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(4,), activation='relu'), 
    tf.keras.layers.Dense(10, activation='relu'),                    
    tf.keras.layers.Dense(3, activation='softmax')                   
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.1)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test set accuracy: {accuracy * 100:.2f}%")
