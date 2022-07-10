import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Inputs are celsius and outputs are fahrenheit
# Model needs to imitate the f = c x 1.8 + 32 equation
celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
# Define the layers input shape is 1 as inputs are one dimensional array
# Units specifies the number neurons in each layer
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
# Create the model with the created layer
model = tf.keras.Sequential([l0])
# Need to measure how well model predicts the outputs this uses a loss function
# Then optimizer will correct model to minimise the loss function
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
# Provides model with input and outputs
# Epochs specify how many times the cycle should be run
# Does the 7 x 500 times
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
# Plots loss over time the models gets better at predicting the output
plt.plot(history.history['loss'])
# Model gets close to correct
print(model.predict([100.0]))
plt.show()