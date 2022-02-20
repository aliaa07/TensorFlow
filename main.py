import tensorflow as tf
import tensorflow.keras as tk
import numpy as np


model = tk.models.Sequential([tk.layers.Dense(500),
                              tk.layers.Dense(500),
                              tk.layers.Dense(1000),
                              tk.layers.Dense(500),
                              tk.layers.Dense(1)])
model.compile(optimizer=tk.optimizers.Adam(lr=0.001), loss=tk.losses.mae, metrics=["mae"])
model.fit(tf.expand_dims(
    np.array([i for i in range(1, 10 ** 5)]), axis=-1),
    np.array([i * 2 for i in range(1, 10 ** 5)]),
    epochs=50)
print(model.predict([30]))
