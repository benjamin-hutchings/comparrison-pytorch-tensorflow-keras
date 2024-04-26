# Centralised asset management for the Streamlit app
class AssetManager:
    def __init__(self):
        # Initialize all asset URLs upon creating an instance of AssetManager
        self.pytorch_logo = "https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"
        self.tensorflow_logo = "https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg"
        self.keras_logo = "https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg"
        self.mnist_gif = "https://upload.wikimedia.org/wikipedia/commons/a/aa/50_mnist_epochs.gif"

    def get_asset(self, asset):
        # Method to return the logo based on the framework name
        if asset.lower() == 'pytorch':
            return self.pytorch_logo
        elif asset.lower() == 'tensorflow':
            return self.tensorflow_logo
        elif asset.lower() == 'keras':
            return self.keras_logo
        elif asset.lower() == 'mnist_gif':
            return self.mnist_gif
        else:
            raise ValueError("Asset not recognized")
        
        