# Centralised asset management for the Streamlit app
class AssetManager:
    def __init__(self):
        # Initialize all asset URLs upon creating an instance of AssetManager
        self.pytorch_logo = "https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"
        self.tensorflow_logo = "https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg"
        self.keras_logo = "https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg"
        self.mnist_gif = "https://upload.wikimedia.org/wikipedia/commons/a/aa/50_mnist_epochs.gif"
        self.neural_net = "https://upload.wikimedia.org/wikipedia/commons/3/3d/Neural_network.svg"

    def add_footer(self):
        import streamlit as st
        # Add footer with Markdown
        st.markdown("---")
        st.markdown("Produced by Benjamin Hutchings | Reach out via [Email](https://elegant-pasca-f13e12.netlify.app/#contact) or [LinkedIn](https://www.linkedin.com/in/benjaminhutchings1/)!")