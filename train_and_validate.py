import numpy as np
import time

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense, Input, concatenate
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.metrics import MeanSquaredError

# size of input and output vectors
ip_size = 466616
op_size = 96

# uniform weight initializer
n = 289 + 290 # 289 training samples and 290 test samples
w_i = 6.0/np.sqrt(n)
print("w_i = ", w_i)
initializer = RandomUniform(minval=-w_i, maxval=w_i)

# train split
set_split = int((n*0.85))

# Define model using Keras' Functional API

branch_ip = Input(shape=(ip_size,))
branch1 = Dense(32, activation = tensorflow.sin, kernel_initializer=initializer)(branch_ip)
branch2 = Dense(32, activation = tensorflow.sin, kernel_initializer=initializer)(branch1)
branch3 = Dense(32, activation = tensorflow.sin, kernel_initializer=initializer)(branch2)
branch4 = Dense(32, activation = tensorflow.sin, kernel_initializer=initializer)(branch3)
branch5 = Dense(32, activation = tensorflow.sin, kernel_initializer=initializer)(branch4)
branch_op = Dense(96, activation = 'relu', kernel_initializer=initializer)(branch5)


model = Model(inputs=branch_ip, outputs=branch_op)
model.summary()

batch_size = 289

class PreprocessedDataset(tensorflow.data.Dataset):
    def _generator(num_samples):
        x_file = open("X_train.csv")
        y_file = open("y_train.csv")
        
        X = np.zeros((batch_size, ip_size), dtype = "float64")
        Y = np.zeros((batch_size, op_size), dtype = "float64")
        
        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            for position, line in enumerate(x_file):
                if position == sample_idx:
                    x = line
                    x = np.fromstring(x, sep=',')
                    X[sample_idx, :] = x
            
            for position, line in enumerate(y_file):
                if position == sample_idx:
                    y = line
                    y = np.fromstring(y, sep=',')
                    Y[sample_idx, :] = y
            
            yield X, Y

    def __new__(cls, num_samples=batch_size):
        return tensorflow.data.Dataset.from_generator(
            cls._generator,
            output_signature = (tensorflow.TensorSpec(shape=(batch_size, ip_size,),  dtype = tensorflow.float32),
                                  tensorflow.TensorSpec(shape=(batch_size, op_size,), dtype = tensorflow.float32)),
            args=(num_samples,)
        )

loss_object = tensorflow.keras.losses.MeanAbsoluteError()
def loss(model, x, y, training):
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
    with tensorflow.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = keras.optimizers.Adam(0.01)

def benchmark(dataset, num_epochs=50):
    start_time = time.perf_counter()
    
    train_loss_results = []
    train_accuracy_results = []
    
    for epoch in range(num_epochs):
        
        epoch_loss_avg = tensorflow.keras.metrics.Mean()
        epoch_accuracy = tensorflow.keras.metrics.MeanAbsoluteError()

        # Training loop - using batches of 32
        for x,y in dataset:
                        
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 5 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, MAE: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
    print("Execution time:", time.perf_counter() - start_time)


def cpu():
    benchmark(
    PreprocessedDataset()
    .prefetch(tensorflow.data.AUTOTUNE)
        )

cpu()

# Validate model

x_file = open("X_test.csv")
y_file = open("y_test.csv")
        
X = np.zeros((batch_size, ip_size), dtype = "float64")
Y = np.zeros((batch_size, op_size), dtype = "float64")
        
for sample_idx in range(291):
    # Reading data (line, record) from the file
    for position, line in enumerate(x_file):
        if position == sample_idx:
            x = line
            x = np.fromstring(x, sep=',')
            X[sample_idx, :] = x
            
    for position, line in enumerate(y_file):
        if position == sample_idx:
            y = line
            y = np.fromstring(y, sep=',')
            Y[sample_idx, :] = y
            
y_hat = model.predict(X)
res = y_hat - Y

# Average MAE on each image

av_mae = []
for i in range(0, len(res), 1) :
    av_mae.append(np.mean(res[i]))
av_mae