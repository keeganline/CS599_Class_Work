from pickle import TRUE
import time

import numpy as np
import tensorflow as tf
###import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

tf.random.set_seed(7510110110397110)

NUM_EXAMPLES = 500
X = tf.random.normal([NUM_EXAMPLES])  # Input values
noise = tf.random.normal([NUM_EXAMPLES])  # Random noise
y = X * 3 + 2 + (noise)  # Target values with noise
loss_type = "MSE"
patience = 300            # Number of steps to wait before reducing LR
lr_decay_factor = 0.5     # Factor to reduce LR by (e.g., multiply by 0.5)
best_loss = float('inf')  # Initialize best loss as infinity
patience_counter = 0
save_weights = []
save_biases =  []     # Counter to track how long loss has not improved

np.random.seed(75101101)

W = tf.Variable(np.random.normal(0)) # Initializing W
b = tf.Variable(np.random.normal(0))  # Initializing b

train_steps = 1000 # Number of training iterations
learning_rate = 0.001  # Step size

start_time = time.process_time()
for i in range(train_steps):
    with tf.GradientTape() as tape:
        # Forward pass: compute predicted y (yhat)
        yhat = X * W + b

        # Compute loss based on the selected loss function
        if loss_type == "MSE":
            loss = tf.reduce_mean(tf.square(yhat - y))
        elif loss_type == "MAE":
            loss = tf.reduce_mean(tf.abs(yhat - y))
        elif loss_type == "Hybrid":
            # Here, we blend L1 and L2 losses.
            # alpha controls the trade-off between L1 (MAE) and L2 (MSE) components.
            alpha = 0.5
            loss = tf.reduce_mean(alpha * tf.abs(yhat - y) + (1 - alpha) * tf.square(yhat - y))
        else:
            raise ValueError("Unknown loss type selected. Choose 'MSE', 'MAE', or 'Hybrid'.")

    # Compute gradients of loss with respect to W and b
    dW, db = tape.gradient(loss, [W, b])

    save_weights.append(W.numpy())
    save_biases.append(b.numpy())
    # Update parameters using gradient descent
    W.assign_sub(learning_rate * dW)
    b.assign_sub(learning_rate * db)

    # --- Learning Rate Scheduling ---
    # If the loss improves, reset the patience counter; otherwise, increase it.
    current_loss = loss.numpy()
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
    else:
        patience_counter += 1

    # If the loss hasn't improved for 'patience' steps, reduce the learning rate.
    if patience_counter >= patience:
        learning_rate *= lr_decay_factor
        print(f"Reducing learning rate to {learning_rate:.6f} at step {i}")
        patience_counter = 0  # Reset the counter after reducing LR

    # Print training progress every 500 steps
    if i % 500 == 0:
        print(f"Step {i}, Loss: {current_loss:.4f}, W: {W.numpy():.4f}, b: {b.numpy():.4f}")

    #if i % 500 == 0:
    #    W.assign_sub(0.05*np.random.normal(0))
    #    b.assign_sub(0.05*np.random.normal(0))

    #if i % 50 == 0:
    #    learning_rate == learning_rate + 0.5*np.random.uniform(0)

print(f"\nFinal Model: W = {W.numpy():.4f}, b = {b.numpy():.4f}, Final Loss: {loss.numpy():.4f}")
end_time = time.process_time()
cpu_time = end_time - start_time
print(f"CPU time: {cpu_time/1000} seconds")


plt.plot(save_biases, label = "Bias")
plt.plot(save_weights, label = "Weight")
plt.legend()
plt.show()


