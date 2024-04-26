# Centralised asset management for the Streamlit app

# Image editing tools
import requests
from PIL import Image
from io import BytesIO

class AssetManager:
    def __init__(self):
        # Initialize all asset URLs upon creating an instance of AssetManager
        self.pytorch_logo = "https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"
        self.tensorflow_logo = "https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg"
        self.keras_logo = "https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg"
        self.mnist_gif = "https://upload.wikimedia.org/wikipedia/commons/a/aa/50_mnist_epochs.gif"
        self.neural_net = "https://upload.wikimedia.org/wikipedia/commons/3/3d/Neural_network.svg"
