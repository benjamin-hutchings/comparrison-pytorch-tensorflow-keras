import streamlit as st
from assets.asset_manager import *

assets = AssetManager()

st.title("Code Comparison")
st.write(
    """
This section compares the implementation of a simple neural network using three major frameworks: PyTorch, TensorFlow, and Keras. 
The comparison covers everything from data loading to model evaluation.

##### Use the navigation pane on the left to see different sections of code!

*The full Jupyter notebooks are available on my [GitHub](https://github.com/benjamin-hutchings/comparrison-pytorch-tensorflow-keras).* 

"""
)

st.write("---")

st.sidebar.title("Code Blocks")
sections = [
    "Imports",
    "Data Loading",
    "Data Visualisation",
    "Model Building",
    "Training",
    "Evaluation",
]
selected_section = st.sidebar.selectbox("Choose a topic", sections)

if selected_section == "Imports":
    st.header("Imports")
    st.write(
        """
- **PyTorch:** Directly imports specific modules for data manipulation, models, and utilities.
- **TensorFlow:** Imports are more generalized, typically only needing the top-level TensorFlow and Keras APIs.
- **Keras:** Focuses on high-level model components, reflecting its streamlined approach to neural network design.
"""
    )

    st.write("### PyTorch Imports")
    st.code(
        """
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split""",
        language="python",
    )

    st.write("### TensorFlow Imports")
    st.code(
        """
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models, utils""",
        language="python",
    )

    st.write("### Keras Imports")
    st.code(
        """
    from tensorflow import keras
    from keras import layers, models""",
        language="python",
    )

elif selected_section == "Data Loading":
    st.header("Data Loading")
    st.write("""
- **PyTorch:** Employs a modular approach where data loading and transformations are explicitly defined and linked.
- **TensorFlow:** Data loading is often integrated with preprocessing steps in a more concise format using TensorFlow's built-in functions.
- **Keras:** Generally utilizes the simplest and most abstracted form of data loading among the three, demonstrating its user-friendliness.
""")
    
    st.write("### PyTorch Data Loading")
    st.code(
        """
            # Load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # Transformation pipeline: converts to pytorch tensor, and normalises to mean and standard deviation of 0.5 & 0.5
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) # Accesses and stores dataset via API, downloads if not available, applies the transform defined above
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) # Same but train=False so that the test dataset is used

    # Code to handle validation split
    total_train_samples = len(train_dataset)
    val_size = int(total_train_samples * 0.2)  # 20% for validation
    train_size = total_train_samples - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders for training, validation, and test subsets
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 
            """,
        language="python",
    )

    st.write("### TensorFlow Data Loading")
    st.code(
        """
            # Load MNIST data
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], as_supervised=True, shuffle_files=True)

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, label # Normalize images to have a value between 0 and 1 and convert to float32

    batch_size = 32  # Define the batch size to be used in training and validation

    ds_train = ds_train.map(normalize_img).shuffle(10000) # Normalize and shuffle the full training dataset with a buffer size of 10,000

    # Calculate the number of total training examples to find out 20% of it for validation
    num_train_examples = ds_train.cardinality().numpy()  # Get the total number of training examples
    num_val_examples = int(num_train_examples * 0.2)     # Calculate 20% of the total number for validation set

    train_data = ds_train.skip(num_val_examples).batch(batch_size) # Skip the first 20% of data to create the training dataset and batch it
    val_data = ds_train.take(num_val_examples).batch(batch_size) # Take the first 20% of the shuffled dataset for validation and batch it

    test_data = ds_test.map(normalize_img).batch(batch_size) # Apply normalization to test data and batch it
            """,
        language="python",
    )

    st.write("### Keras Data Loading")
    st.code(
        """
            # Load data
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data() # keras API used to access MNIST dataset, two tuples returned
    train_images = train_images.reshape((60000, 784)).astype('float32') / 255 # 60000 images flattened to 1-D 784 pixels long and converted to floats, then normalised between [0-1]
    test_images = test_images.reshape((10000, 784)).astype('float32') / 255 # reshape is a numpy function but accessible via tf
    train_labels = utils.to_categorical(train_labels) # labels converted to binary vectors e.g. 3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], because categorical crossentropy outputs probabilities across classes
    test_labels = utils.to_categorical(test_labels)
            """,
        language="python",
    )

elif selected_section == "Data Visualisation":
    st.header("Data Visualisation")
    st.write("""
  - PyTorch's visualisation approach is highly customizable, requiring manual setup for image unnormalization and conversion of tensors to NumPy arrays for plotting. The use of `matplotlib` for visualization is straightforward but requires more code to handle the data preprocessing and setup.
  - TensorFlow integrates smoothly with `matplotlib` to visualize data directly from tensors.
  - Keras,  focuses on high-level functions, demonstrating its user-friendly approach. The visualization code in Keras closely resembles that in TensorFlow but with slight variations that reflect Keras' streamlined API.
    """)
    
    st.write("### PyTorch Data Visualisation")
    st.code(
        """
            def imshow(imgs):
        imgs = imgs / 2 + 0.5  # Unnormalize the images: revert the earlier normalization by re-scaling them to [0,1]
        npimgs = imgs.numpy()  # Convert the tensor to a NumPy array for plotting
        plt.imshow(np.transpose(npimgs, (1, 2, 0)))  # Transpose the dimensions from [C, H, W] to [H, W, C] and display image
        plt.axis('off')  # Hide the axes on the plot
        plt.show()  # Display the imageW

    dataiter = iter(train_loader) # Create an iterator from the training DataLoader
    images, labels = next(dataiter) # Fetch the next batch of images and labels

    img_grid = utils.make_grid(images[0:24], nrow=6)  # Create a grid of images using the first 24 images from the batch
    imshow(img_grid) # Display the image grid
            """,
        language="python",
    )

    st.write("### TensorFlow Data Visualisation")
    st.code(
        """
            ### Optional code to plot examples of the images
    # Function to plot a grid of images
    def plot_images_grid(dataset, num_rows=4, num_cols=5):
        plt.figure(figsize=(7, 5)) # larger figure size for better visibility
        for images, labels in dataset.take(1):  # Take one batch from the dataset
            for i in range(num_rows * num_cols):
                plt.subplot(num_rows, num_cols, i+1)
                plt.imshow(images[i].numpy().reshape(28, 28), cmap='gray')  # Reshape and plot image
                plt.title(f'Label: {labels[i].numpy()}')  # Display the label
                plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Call the function with the training data
    plot_images_grid(train_data)
            """,
        language="python",
    )

    st.write("### Keras Data Visualisation")
    st.code(
        """
            # Requires numpy
    import numpy as np

    # Function to plot a grid of images
    def plot_images_grid(data, labels, num_rows=4, num_cols=5): # default [4x5] subplots in grid
        plt.figure(figsize=(8,5)) # image width and height
        for i in range(num_rows * num_cols): # loop through total imgs
            plt.subplot(num_rows, num_cols, i+1) # subplot index dependent on loop iteration
            plt.imshow(data[i].reshape(28,28), cmap='gray') # data[i] is index in dataset, reshape used as the images were flattened in pre-processing
            plt.title(f'Label: {np.argmax(labels[i])}') # corresponding title as dataset label
            plt.axis('off') # no borders or axis values for clarity
        plt.tight_layout() # no overlap between images
        plt.show() # render plot
        
    plot_images_grid(train_images, train_labels)
            """,
        language="python",
    )

elif selected_section == "Model Building":
    # Model Building section
    st.header("Model Building")
    st.write("""
- **PyTorch:** Requires explicit definition of the model architecture and forward pass, offering granular control.
- **TensorFlow:** Provides flexibility in model definition, either through sequential or functional APIs, suitable for both simple and complex models.
- **Keras:** Uses a highly abstracted method, making it the most straightforward for beginners to define models quickly.
""")
    
    st.write("### PyTorch Model Building")
    st.code(
        """# Every model in PyTorch is implemented using a class that inherits from nn.Module
    class Net(nn.Module):
        def __init__(self):  # Constructor function to initialize the neural network
            super(Net, self).__init__()  # Call to the parent class (nn.Module) constructor to handle the underlying initialization
            self.fc1 = nn.Linear(784, 128)  # Define the first fully connected (dense) layer with 784 inputs and 128 outputs
            self.fc2 = nn.Linear(128, 64)   # Define the second fully connected layer with 128 inputs and 64 outputs
            self.fc3 = nn.Linear(64, 10)    # Define the third fully connected layer with 64 inputs and 10 outputs, for 10 class outputs

        def forward(self, x):  # Define the forward pass of the network, which outlines how the input 'x' flows through the network
            x = torch.relu(self.fc1(x))  # Apply the ReLU activation function to the output of the first layer
            x = torch.relu(self.fc2(x))  # Apply the ReLU activation function to the output of the second layer
            x = torch.softmax(self.fc3(x), dim=1)  # Apply the softmax activation function to the output of the third layer to normalize the output to a probability distribution over predicted output classes
            return x  # Return the final output of the network
        
        # This is not required to train the model, but is when loading the model for deployment!
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
            def serve(self, x):
            return self(x)
        
    model = Net() # add '.to(device)' for hardware targeting (e.g., model.to('cuda') to run the model on a GPU)
    """,
        language="python",
    )

    st.write("### TensorFlow Model Building")
    st.code(
        """# Define the model (as a class inheriting from tf.Module)
    class SimpleNN(tf.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()  # Call the initializer of the parent class tf.Module
            # Initialize the first layer's weights and biases with random values
            self.weights1 = tf.Variable(tf.random.normal([784, 128], stddev=0.1))
            self.biases1 = tf.Variable(tf.zeros([128]))
            # Initialize the second layer's weights and biases with random values
            self.weights2 = tf.Variable(tf.random.normal([128, 10], stddev=0.1))
            self.biases2 = tf.Variable(tf.zeros([10]))

        def __call__(self, x):
            x = tf.reshape(x, [-1, 784])  # Flatten the image from a 28x28 matrix to a 784-element vector
            x = tf.add(tf.matmul(x, self.weights1), self.biases1)  # Compute the output of the first layer
            x = tf.nn.relu(x)  # Apply ReLU activation function to the first layer's output
            x = tf.add(tf.matmul(x, self.weights2), self.biases2)  # Compute the output of the second layer
            return x  # Return the final output of the network

    # Create an instance of the SimpleNN class
    model = SimpleNN()""",
        language="python",
    )

    st.write("### Keras Model Building")
    st.code(
        """model = models.Sequential([ 
        layers.Flatten(input_shape=(784,)), # Converts the input to a 1D array (28x28=784) (not technically required given our pre-processing step, but good to present the process)
        layers.Dense(128, activation='relu'), # Fully connected layer with 128 neurons, each with a Rectified Linear Unit (ReLU) activation function
        layers.Dense(64, activation='relu'), # Fully connected layer with 64 neurons, each with a Rectified Linear Unit (ReLU) activation function
        layers.Dense(10, activation='softmax') # Fully connected layer with 10 neurons (for 10 output classes), each with a Softmax activation function to output a probability betwen [0-1]
    ])
    """,
        language="python",
    )

elif selected_section == "Training":
    st.header("Training")
    st.write("""
- **PyTorch:** The training loop requires manual setup of both the loop and explicit backpropagation, offering high flexibility and control over the training process.
- **TensorFlow:** Utilizes context managers for gradient calculation, which can make the code cleaner but slightly less transparent than PyTorch's explicit approach.
- **Keras:** The training process is abstracted to a single function call (`fit`), making it extremely user-friendly but at the cost of customizability.
""")
    
    st.write("### PyTorch Training Code")
    st.code(
        """
            # Initialise the optimiser, and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) # have to use an object from the API class for the optimiser
    criterion = nn.CrossEntropyLoss()

            # Training loop
    model.train()  # Ensure the model is in training mode

    # Lists to store training and validation metrics after each epoch
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Iterate over the number of epochs
    for epoch in range(10):
        # Initialize metrics for each epoch
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Ensure the model is in training mode
        model.train()

        # Iterate over each batch of data
        for data, target in train_loader:
            # Reshape or flatten the data as required by the model
            data = data.view(data.shape[0], -1)  # Flatten the images

            # Zero the gradients from previous iterations (pytorch accumlates gradients on tensoirs from the backpass by default)
            optimizer.zero_grad() # (stops the last gradient(s) calculated in backprop being added, stoping multiple gradients accumulating)

            # Forward pass: 
            output = model(data) # Compute predicted outputs by passing inputs to the model
            loss = criterion(output, target) # Calculate the batch loss using the loss criterion

            # Perform backpropagation
            loss.backward() # Compute gradient of the loss with respect to model parameters
            optimizer.step() # Perform a single optimization step (parameter update)

            # Update training loss and accuracy calculations
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        # Record the average training loss and accuracy for the current epoch
        train_losses.append(train_loss / train_total)
        train_accuracies.append(100 * train_correct / train_total)

        # Validation phase
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Set the model to evaluation mode
        model.eval()

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for data, target in val_loader:
                # Reshape or flatten data
                data = data.view(data.shape[0], -1)

                # Forward pass 
                output = model(data) # Compute predicted outputs by passing inputs to the model
                loss = criterion(output, target)# Calculate the batch loss

                # Update validation loss and accuracy calculations
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        # Record the average validation loss and accuracy for the current epoch
        val_losses.append(val_loss / val_total)
        val_accuracies.append(100 * val_correct / val_total)

        # Print training/validation statistics
        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, '
            f'Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.2f}%')

    # Indicate that training is finished
    print('Finished Training.')
            """,
        language="python",
    )

    st.write("### TensorFlow Training Code")
    st.code(
        """
            # Define loss function
    def loss_fn(logits, labels):
        # Use sparse softmax cross entropy with logits to calculate the loss
        # This function calculates softmax cross-entropy between logits and labels
        # It applies softmax on logits internally and handles the numerical instability
        # logits are the raw, unnormalized scores output by the last layer of the network
        # labels are the true labels associated with the data
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # Define optimizer
    learning_rate = 0.001
    optimizer = tf.optimizers.Adam(learning_rate)

    # Function to compute accuracy based on logits and true labels
    def compute_accuracy(logits, labels):
        predictions = tf.argmax(logits, axis=1, output_type=tf.int64)  # Determine the predicted class from logits
        return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))  # Calculate mean accuracy across the batch

    # Lists to store metrics over epochs
    train_losses = []  # Store training loss per epoch
    train_accuracies = []  # Store training accuracy per epoch
    val_losses = []  # Store validation loss per epoch
    val_accuracies = []  # Store validation accuracy per epoch

    # Training step function to perform a single step of the training process
    def train_step(images, labels):
        with tf.GradientTape() as tape:  # Context manager to record operations for automatic differentiation
            logits = model(images)  # Compute the logits (model outputs before activation function) by passing images to the model
            loss = loss_fn(logits, labels)  # Compute the loss by comparing the logits to the true labels
        gradients = tape.gradient(loss, model.trainable_variables)  # Compute gradients of the loss w.r.t. model variables
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Apply gradients to variables (gradient descent step)
        return loss  # Return the computed loss to track progress

    # Validation step function to evaluate the model on the validation dataset
    def validate_step(dataset):
        total_loss = 0  # Initialize total loss for the dataset
        total_accuracy = 0  # Initialize total accuracy for the dataset
        num_batches = 0  # Counter for batches processed
        for images, labels in dataset:
            logits = model(images)  # Compute logits for a batch
            loss = loss_fn(logits, labels)  # Compute loss for a batch
            accuracy = compute_accuracy(logits, labels)  # Compute accuracy for a batch
            total_loss += loss.numpy()  # Aggregate loss over all batches
            total_accuracy += accuracy.numpy()  # Aggregate accuracy over all batches
            num_batches += 1  # Increment batch counter
        avg_loss = total_loss / num_batches  # Calculate average loss per batch
        avg_accuracy = total_accuracy / num_batches  # Calculate average accuracy per batch
        return avg_loss, avg_accuracy  # Return average loss and accuracy

    epochs = 10  # Define the number of epochs for training

    # Loop over epochs for training and validation
    for epoch in range(epochs):
        total_train_loss = 0  # Sum of training losses for epoch
        total_train_accuracy = 0  # Sum of training accuracies for epoch
        num_train_batches = 0  # Number of training batches processed

        # Iterate over batches in the training dataset
        for images, labels in train_data:
            loss = train_step(images, labels)  # Perform a training step and get the loss
            total_train_loss += loss.numpy()  # Aggregate training loss
            accuracy = compute_accuracy(model(images), labels)  # Compute training accuracy for the batch
            total_train_accuracy += accuracy.numpy()  # Aggregate training accuracy
            num_train_batches += 1  # Count training batches

        # Store and print epoch metrics for training
        train_losses.append(total_train_loss / num_train_batches)
        train_accuracies.append(total_train_accuracy / num_train_batches)

        # Perform validation after each epoch and store validation results
        val_loss, val_accuracy = validate_step(val_data)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print results for the epoch
        print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Train Accuracy: {train_accuracies[-1]}, '
            f'Validation Loss: {val_losses[-1]}, Validation Accuracy: {val_accuracies[-1]}')

    print('Training complete.')

            """,
        language="python",
    )

    st.write("### Keras Training Code")
    st.code(
        """
        # Compile the model
    model.compile(optimizer="adam", # Set optimiser
                loss="categorical_crossentropy", # Set loss function
                metrics=["accuracy"]) # List of metrics to track during training
                
    # Train the model
    history = model.fit(train_images, train_labels, # history object used to access training stats
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)

    # .fit() method is used for (not limited to):
    #   - Callbacks (functions applied during training e.g. early stopping)
    #   - Validation tuples (define specific val data)
    #   - Class weighting
    #   - Sample weighting
    #   - etc.
                """
    )

elif selected_section == "Evaluation":
    st.header("Evaluation")
    st.write("""
    ### Metrics Visualisation
- **PyTorch:** Uses utilities from `torchvision` and `matplotlib` to visualize images in a grid, demonstrating flexibility in handling and displaying image data.
- **TensorFlow:** Visualization in TensorFlow often integrates with other Python libraries like `matplotlib`, showing its compatibility with the wider Python ecosystem.
- **Keras:** Since Keras is built on top of TensorFlow, it shares similar visualization capabilities but often simplifies the process by integrating more directly with TensorFlow's functionalities.

### Test Evaluation
- **PyTorch:** Evaluation requires manually setting the model to evaluation mode and disabling gradient computations, which is essential for performance optimization.
- **TensorFlow:** Similar to training, uses a more integrated approach for evaluation, relying on TensorFlow's metrics and session-based computations.
- **Keras:** Provides built-in methods for evaluating the model, which are easy to use and require minimal code, reflecting Keras's aim to simplify the user experience.
""")    
    
    st.write("### PyTorch Evaluation")
    st.code(
        """
            # Plotting the training and validation statistics
    plt.figure(figsize=(15, 6))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Evaluation
    model.eval() # switch to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad(): # no gradient tracking
        for data, target in test_loader:
            data = data.view(data.shape[0], -1)
            
            outputs = model(data)
            
            # Take class labels with highest probability prediction
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    print(f'Accuracy: {100 * correct / total}%')
            """,
        language="python",
    )

    st.write("### TensorFlow Evaluation")
    st.code(
        """
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Evaluation function
    def evaluate(dataset):
        accuracy_metric = tf.metrics.Accuracy()
        for images, labels in dataset:
            logits = model(images)
            predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
            accuracy_metric.update_state(labels, predictions)
        return accuracy_metric.result().numpy()

    # Evaluate the model
    accuracy = evaluate(test_data)
    print(f'Test accuracy: {accuracy}%')
    """,
        language="python",
    )

    st.write("### Keras Evaluation")
    st.code(
        """
        # Visualise model stats
    plt.figure(figsize=(15, 6))  # Width, Height in inches

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    # Display the plots
    plt.tight_layout()
    plt.show()

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc * 100}%')
    """,
        language="python",
    )

    st.write(
        "If you've made it this far, it's time to look at some stats and try the models for yourself"
    )

# Footer
assets.add_footer()