import streamlit as st
from keras.models import load_model
import tensorflow as tf
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io
from assets.asset_manager import AssetManager

assets = AssetManager()

# Load Keras model
keras_model = load_model('streamlit_app/assets/trained_models/keras_model.keras')

# Define and load the PyTorch model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

torch_model = Net()
torch_model.load_state_dict(torch.load('streamlit_app/assets/trained_models/pytorch_model_state_dict.pth'))
torch_model.eval()

# Load TensorFlow model
@st.cache_resource
def load_tf_model():
    model_path = 'streamlit_app/assets/trained_models/tf_model'
    model = tf.saved_model.load(model_path)
    return model.signatures['serving_default']

tf_model = load_tf_model()

# Image preprocessing for each model
def preprocess_image(image):
    image = image.resize((28, 28)).convert('L')
    image_array = np.array(image)
    return {
        'keras': np.expand_dims(image_array / 255.0, axis=0).flatten().reshape(1, 784),
        'pytorch': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])(image),
        'tensorflow': image_array.astype(np.float32) / 255.0
    }

# Predictions
def predict(image):
    processed_images = preprocess_image(image)
    keras_pred = keras_model.predict(processed_images['keras'])
    keras_label = np.argmax(keras_pred)
    
    with torch.no_grad():
        pt_pred = torch_model(processed_images['pytorch'].unsqueeze(0))
    pt_label = torch.argmax(pt_pred, dim=1)

    tf_pred = tf_model(tf.constant(processed_images['tensorflow'].reshape(1, 784)))
    tf_label = np.argmax(tf_pred['output_0'])

    return keras_label, pt_label.item(), tf_label

# Streamlit app setup
st.title('Try the models out yourself!')
st.write('''Upload an image, or use one of the sample MNIST images below for classification.
         Then press the 'Classify' button below!
         ''')

# Display sample MNIST images with download links
st.write("## MNIST Sample Images")
mnist_cols = st.columns(5)
mnist_images = ['streamlit_app/assets/mnist_samples/mnist_sample_0.png',
                'streamlit_app/assets/mnist_samples/mnist_sample_1.png',
                'streamlit_app/assets/mnist_samples/mnist_sample_4.png',
                'streamlit_app/assets/mnist_samples/mnist_sample_5.png',
                'streamlit_app/assets/mnist_samples/mnist_sample_9.png']

for col, img_path in zip(mnist_cols, mnist_images):
    with col:
        image = Image.open(img_path)
        st.image(image, use_column_width=True)
        with open(img_path, "rb") as file:
            btn = st.download_button(
                label="Download",
                data=file,
                file_name=img_path.split('/')[-1],
                mime="image/png"
            )

# Upload and classify UI section
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption='Uploaded Image', use_column_width=True, width=300)  # Adjusted image size for consistency

    if st.button('Classify'):
        with st.spinner('Classifying...'):
            keras_label, pt_label, tf_label = predict(image)
            st.success('Classification complete.')

        # Displaying the logos and predictions in a grid layout
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.image(assets.pytorch_logo, caption='PyTorch', width=100)
        with col2:
            st.image(assets.tensorflow_logo, caption='TensorFlow', width=120)
        with col3:
            st.image(assets.keras_logo, caption='Keras', width=125)
            
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown(f"**Prediction:** {pt_label}")
        with col2:
            st.markdown(f"**Prediction:** {tf_label}")
        with col3:
            st.markdown(f"**Prediction:** {keras_label}")
            
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.text("PyTorch offers high-speed\ncustom model training.")
        with col2:
            st.text("TensorFlow provides robust\nscalability in production.")
        with col3:
            st.text("Keras simplifies the\ncreation of deep models.")

        
        st.write("## Analysis")
        # Example to insert a GIF based on model's performance
        if keras_label == pt_label == tf_label:
            st.write("Hooray!!")
            st.image("https://www.lolgifs.net/wp-content/uploads/2019/01/its-time-to-party.gif", caption="Great Consistency!")
        else:
            st.info("The models disagree, showing the diversity in model training approaches.")
        
        st.write("### Congratulations on utilising an end-to-end machine learning application!")