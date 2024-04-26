import streamlit as st

# Page Title
st.title('Neural Network Model Overview')

# Introduction to the Model
st.header('Model Architecture')
st.write("""
The model used for this comparison is a simple fully connected neural network, often referred to as a Multilayer Perceptron (MLP). 
It consists of three layers:
- **Input Layer:** Takes the flattened MNIST images as input (784 features, since each image is 28x28).
- **Hidden Layers:** Two layers with ReLU activations to introduce non-linearity, facilitating the model's learning of complex patterns in the data.
- **Output Layer:** A softmax layer that outputs the probability distribution across the 10 digit classes.
""")

# Model Diagram or Code
st.code("""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)   # Second fully connected layer
        self.fc3 = nn.Linear(64, 10)    # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x
""", language='python')

# Explanation of Dataset
st.header('MNIST Dataset')
st.write("""
The MNIST dataset, which stands for Modified National Institute of Standards and Technology database, is a large database of handwritten digits that is commonly used for training various image processing systems. The dataset contains 60,000 training images and 10,000 testing images, each of which is a 28x28 pixel grayscale image of a single handwritten digit.
""")

# Experiment Setup
st.header('Experiment Setup')
st.write("""
For a fair comparison across the different frameworks, each implementation must adhere to the following criteria:
- **Data Handling:** Load the MNIST dataset, normalize the data, and split it into training, validation (20% of training data), and testing sets.
- **Training:** Train the model for 10 epochs, recording the training and validation losses and accuracies after each epoch.
- **Evaluation:** Evaluate the model on the test data to measure its performance.
""")

# Additional Implementation Details
st.subheader('Implementation Details')
st.write("""
Each framework (PyTorch, TensorFlow, and Keras) will implement the same model architecture and training regimen to ensure that comparisons are based on equivalent bases. The focus will be on how each framework handles data loading, model building, training, and evaluation.
""")

# Conclusion and Navigation
st.write("""
Navigate to other pages in this application to explore the code comparison, run simulations, and view performance metrics for each framework.
""")

# Footer
st.markdown("---")
st.markdown("Produced by Benjamin Hutchings | Contact me via Email or LinkedIn!")