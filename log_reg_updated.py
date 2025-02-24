import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist # type: ignore
import time 


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize data

class LogisticRegressionModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.W = tf.Variable(tf.zeros([28*28, 10]))  # 10 classes
        self.b = tf.Variable(tf.zeros([10]))

    def __call__(self, x):
        return tf.nn.softmax(tf.matmul(x, self.W) + self.b)  # Softmax activation

# Model instance
model = LogisticRegressionModel()

# Train Model 
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred))

def train_step(model, images, labels, optimizer, lambda_reg=0.0):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, images, labels, lambda_reg)
    # Compute gradients with respect to the model's variables.
    grads = tape.gradient(loss, [model.W, model.b])
    # Update the variables using the optimizer.
    optimizer.apply_gradients(zip(grads, [model.W, model.b]))
    return loss 


# Compute Loss
def compute_loss(model, images, labels, lambda_reg=0.0):
    predictions = model(images)
    # Compute the standard sparse categorical cross-entropy loss.
    ce_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(labels, predictions))
    # L2 loss on weights (do not regularize bias).
    l2_loss = tf.nn.l2_loss(model.W)  # Sum of squares divided by 2
    return ce_loss + lambda_reg * l2_loss

# Training and Validation Splits
np.random.seed(123)
val_split = 0.5
num_val = int(train_images.shape[0] * val_split)
val_images = train_images[:num_val]
val_labels = train_labels[:num_val]
train_images_split = train_images[num_val:]
train_labels_split = train_labels[num_val:]

# Build Model
class LogisticRegressionModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.W = tf.Variable(tf.zeros([28 * 28, 10]), name="weights")
        self.b = tf.Variable(tf.zeros([10]), name="biases")

    def __call__(self, x):
        # Cast input to float32 to ensure consistency.
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, [-1, 28 * 28])
        logits = tf.matmul(x, self.W) + self.b
        return tf.nn.softmax(logits)

model = LogisticRegressionModel()

def train_model(optimizer, lambda_reg=0.0, num_epochs=20, batch_size=64):
    # Create a new model instance (fresh initialization)
    model = LogisticRegressionModel()
    # Prepare tf.data datasets for training and validation
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images_split, train_labels_split))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(batch_size)
    # Lists to record metrics for each epoch.
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }
    # For accuracy computation
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(1, num_epochs + 1):
        # Reset metrics at the start of each epoch.
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()

        # Training loop
        epoch_losses = []
        for batch_images, batch_labels in train_dataset:
            loss = train_step(model, batch_images, batch_labels, optimizer, lambda_reg)
            epoch_losses.append(loss.numpy())
            # Update training accuracy.
            predictions = model(batch_images)
            train_acc_metric.update_state(batch_labels, predictions)

        # Compute average training loss over epoch.
        train_loss = np.mean(epoch_losses)
        train_accuracy = train_acc_metric.result().numpy()

        # Validation loop
        val_losses = []
        for batch_images, batch_labels in val_dataset:
            loss = compute_loss(model, batch_images, batch_labels, lambda_reg)
            val_losses.append(loss.numpy())
            predictions = model(batch_images)
            val_acc_metric.update_state(batch_labels, predictions)
        val_loss = np.mean(val_losses)
        val_accuracy = val_acc_metric.result().numpy()

        # Save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch:02d}: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
    return model, history


# Optimizers
optimizers_to_try = {
    "SGD": tf.optimizers.SGD(learning_rate=0.001),
    "Adam": tf.optimizers.Adam(learning_rate=0.001),
    "RMSprop": tf.optimizers.RMSprop(learning_rate=0.001)
}

lambda_reg = 0.001  # Change to 0.0 to see training without regularization.
num_epochs = 15  # Adjust as needed
history_dict = {}

start_time = time.process_time()
for opt_name, opt in optimizers_to_try.items():
    print(f"Training with {opt_name} optimizer (lambda_reg={lambda_reg})")
    # Train a new model for each optimizer.
    history = train_model(opt, lambda_reg=lambda_reg, num_epochs=num_epochs, batch_size=128)[1]
    history_dict[opt_name] = history

end_time = time.process_time()
cpu_time = end_time - start_time
print(f"CPU time: {cpu_time/15} seconds")



# Plot History
epochs = range(1, num_epochs + 1)
fig, axs = plt.subplots(2, 1, figsize=(5, 5))
for opt_name, history in history_dict.items():
    axs[0].plot(epochs, history["train_loss"], label=f"{opt_name} Train")
    axs[0].plot(epochs, history["val_loss"], '--', label=f"{opt_name} Val")
axs[0].set_title("Loss over Epochs")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend(fontsize="small")
axs[0].grid(True)

for opt_name, history in history_dict.items():
    axs[1].plot(epochs, history["train_accuracy"], label=f"{opt_name} Train")
    axs[1].plot(epochs, history["val_accuracy"], '--', label=f"{opt_name} Val")
axs[1].set_title("Accuracy over Epochs")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend(fontsize="small")
axs[1].grid(True)

plt.tight_layout()
plt.show()

# test 3x3 plot of images
def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape([28,28]), cmap='binary')

        # Show true and predicted classes.
        if yhat is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], yhat[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

images = test_images[0:9]

# Get the true classes for those images.
y = test_labels[0:9]

# Plot the images and labels using our helper-function above.
#plot_images(images=images, y=y)

# Test accuracy
x = tf.cast(images[0], tf.float32)
x = tf.reshape(x, [-1, 28 * 28])
logits = tf.matmul(x, model.W) + model.b
print(model.W)
print(tf.nn.softmax(logits))
print(y[0])

#random forest 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
#rf.fit(train_images,train_labels)

#ypred = rf.predict(test_images)
#print(ypred)