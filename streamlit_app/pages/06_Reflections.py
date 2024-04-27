import streamlit as st

from assets.asset_manager import *

assets = AssetManager()


st.write("""
         
The three scripts provided each use a different deep learning framework—PyTorch, Keras, and TensorFlow without Keras—to build and train a neural network for classifying the MNIST dataset. Here are the key differences between each approach, along with their strengths and weaknesses:

PyTorch
Key Features:

PyTorch provides a more flexible and dynamic environment for defining models due to its eager execution.
Model building involves defining a class that inherits from nn.Module, providing a clear structure for larger models.
Gradient zeroing (optimizer.zero_grad()) is explicit, helping in understanding the training process.
Strengths:

Flexibility in model design and experimentation.
The manual control over aspects like data loading and gradient management is excellent for educational purposes and complex scenarios.
Weaknesses:

More verbose and requires more boilerplate code compared to Keras, potentially leading to longer development times for simpler projects.
Keras
Key Features:

Keras is highly user-friendly with a simple API, focusing on the ease of use.
The model definition and training process are streamlined, typically requiring less code.
Uses a built-in function for splitting the dataset into training and validation, which simplifies the code further.
Strengths:

Fast prototyping is possible due to the high-level abstraction of complex routines.
Great for beginners and projects that need to be developed quickly.
Integrated support for training metrics and validation.
Weaknesses:

Less flexible in model design compared to PyTorch or TensorFlow's low-level APIs.
Sometimes, debugging and performance tuning can be more challenging due to the abstraction.
TensorFlow (without Keras)
Key Features:

TensorFlow provides a mix of low-level and high-level operations, offering more control over the model than Keras.
The use of TensorFlow's data pipeline (tf.data) for efficient data handling.
Explicit definition of training and validation steps using TensorFlow's lower-level operations.
Strengths:

Highly scalable and suitable for production environments.
Offers detailed control over almost every aspect of the training and model architecture.
Weaknesses:

Can be verbose and complex, making it harder to learn and use compared to Keras.
Requires a good understanding of TensorFlow's API for effective implementation.
General Considerations for a Streamlit WebApp
When integrating these scripts into a Streamlit web app for side-by-side comparison, consider focusing on the following sections for each framework:

Data Loading: Highlight how each handles data transformations and batching.
Model Definition: Compare the verbosity and structural differences.
Training Loop: Discuss the flexibility or simplicity of each approach.
Validation and Metrics Handling: Show how each framework handles validation and metrics calculation.
Evaluation: Compare the final evaluation steps to see how each framework handles testing.
By focusing on these aspects, users can clearly see the differences in approach, complexity, and usability of each framework, helping them decide which one might be best suited for their specific needs.

User
Could you go through the code section by section and compare how each framework is different please?
ChatGPT
Absolutely! Let’s compare how each framework handles various sections of the code used to build and train a neural network for classifying the MNIST dataset. We'll focus on the key parts: Data Loading, Model Definition, Training Loop, Validation, and Evaluation.

1. Data Loading
PyTorch:

Uses torchvision to download and transform datasets.
Handles transformations through a pipeline using transforms.Compose.
Manually splits the dataset into training, validation, and test sets using random_split.
Keras:

Directly loads the MNIST dataset from Keras’ dataset collection.
Manually reshapes and normalizes the data.
Automatically manages training and validation splits during model fitting using validation_split.
TensorFlow:

Loads data using TensorFlow Datasets, which provides a more extensive configuration for shuffling and batching.
Applies transformations using the map method.
Manually calculates and splits the training dataset into training and validation using TensorFlow operations.
Comparison:

PyTorch and TensorFlow offer more manual control over data manipulation and batching, which is useful for custom datasets or specific preprocessing needs.
Keras simplifies data loading and preprocessing, making it less prone to errors and quicker to set up but at the cost of flexibility.
2. Model Definition
PyTorch:

Defines models by creating a class that inherits from nn.Module.
Uses a more explicit and detailed setup, requiring the definition of each layer and the forward pass explicitly.
Keras:

Utilizes the Sequential API to stack layers, making model creation straightforward and concise.
Less verbose, with many defaults set to simplify the process.
TensorFlow:

Similar to PyTorch, TensorFlow (without Keras) requires defining the model more explicitly and manually managing weights and computations.
Uses low-level TensorFlow operations, giving full control over every aspect of the model.
Comparison:

Keras is the easiest in terms of setup and readability, ideal for standard models.
PyTorch and TensorFlow provide a deep level of customization and are better suited for complex, custom architectures that may require fine-tuning and direct manipulation of operations.
3. Training Loop
PyTorch:

Manually iterates over epochs and batches.
Explicitly handles zeroing gradients, performing backpropagation, and updating model parameters.
Keras:

The training loop is abstracted away completely using the fit method, which simplifies the code and handles everything internally including looping through batches and epochs, as well as applying optimizations.
TensorFlow:

Similar to PyTorch, it requires manual setup of the training loop, including handling gradients and applying updates.
Comparison:

Keras significantly simplifies the training process, beneficial for quick implementations and standard use cases.
PyTorch and TensorFlow provide granular control over the training process, which is crucial for research and developing non-standard training dynamics.
4. Validation
PyTorch:

Manually handles the validation within the training loop, setting the model to evaluation mode and turning off gradients.
Keras:

Handles validation automatically when the validation_split or validation_data is specified in the fit method.
TensorFlow:

Similar to PyTorch, requires explicit validation steps, managing running averages of losses and other metrics manually.
Comparison:

Keras offers a much simpler validation integration, reducing code and potential errors.
PyTorch and TensorFlow require manual validation handling, which allows for more customized validation logic.
5. Evaluation
PyTorch:

Manually sets the model to evaluation mode and disables gradient calculations during the evaluation phase.
Keras:

Uses the evaluate method to handle model evaluation, which is straightforward and automatically manages batch processing of test data.
TensorFlow:

Requires manually setting up the evaluation loop, similar to the training loop, and managing metrics.
Comparison:

Keras continues to provide ease with its evaluate method, streamlining the evaluation process.
PyTorch and TensorFlow offer flexibility but require more code to correctly evaluate the model.


What I learn:
Using signatures for saving and loading a tf model without keras
Streamlit app development - data visualisation, interactivity, caching and deployment
some nuaces for .csv -> pandas df requirements for data visualisation

""")


# Footer
assets.add_footer()