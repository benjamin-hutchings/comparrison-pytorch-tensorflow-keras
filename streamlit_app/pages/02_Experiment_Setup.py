import streamlit as st
from assets.asset_manager import *

assets = AssetManager()

# Page Title
st.title('Experiment Setup')

# Introduction to the Model
st.header('Model Architecture')
st.write("""
The model used for this comparison is a simple fully connected neural network, often referred to as a Multilayer Perceptron (MLP). 
It consists of three layers:
- **Input Layer:** Takes the flattened MNIST images as input (784 features, since each image is 28x28).
- **Hidden Layers:** Two layers with ReLU activations to introduce non-linearity, facilitating the model's learning of complex patterns in the data.
- **Output Layer:** A softmax layer that outputs the probability distribution across the 10 digit classes.
""")

st.write("""
Here is the code in PyTorch - don't worry, we'll go over this in the next section!
""")

# Model Code
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

st.write("""
Imagine this neural network below but with two hidden layers, and a few more nodes and connections!
""")

st.image(assets.neural_net, width=250, caption="Multilayer Perceptron Network Example")

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

st.write(" ")
st.image("https://media1.giphy.com/media/3o7TKyH7Ur3kvjGa5y/giphy.gif?cid=6c09b952gfv79hvcvioskbuhw6nrzo40pj06btw3vjls0j5w&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=g", caption="We best be careful... ;)")

# Footer
assets.add_footer()