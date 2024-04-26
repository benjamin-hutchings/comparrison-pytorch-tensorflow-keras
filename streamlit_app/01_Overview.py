import streamlit as st
from assets.asset_manager import AssetManager

assets = AssetManager()

# MNIST gif
st.image(assets.mnist_gif, width=150, caption="MNIST Handwritting recognition dataset")

# Page Title
st.title('Deep Learning Frameworks')

# Introduction
st.write("""
This mini-project compares the use of three major deep learning frameworks: PyTorch, TensorFlow, and Keras. 
By building the same neural network model across these frameworks, we explore how each handles model building, 
training, and evaluation, providing insights into their usability, flexibility, and performance.
""")

# Contents Section
st.header('Contents')
st.write("""
- **Overview**: A brief introduction to each framework and this project.
- **The Neural Network**: Introducting the ANN architecture, the MNIST dataset and key control variables.
- **Code Comparison**: Side-by-side code snippets showing how each framework implements the same model.
- **Statistics**: Comparison of model performance metrics across frameworks.
- **Try the Models!**: Interactive section to try out models and modify parameters.
- **Reflections**: My closing thoughts...
""")

# Overview Section
st.header('Overview')
st.write("""
Deep learning frameworks have dramatically simplified the process of developing and deploying neural network models. 
They vary widely in terms of syntax, ease of use, and the level of abstraction they provide, which can significantly 
affect both the development time and the performance of the models.
""")

# Summary of Pytorch
col1, col2 = st.columns([1,2])
with col1:
    st.image(assets.pytorch_logo, width=150)
with col2:
    st.subheader('PyTorch')
    st.write("""
    PyTorch is known for its flexibility and dynamic computation graph that allows changes to the model architecture on-the-fly.
    - **Dynamic Nature**: Ideal for dynamic input lengths and experimentation.
    - **User Base**: Widely popular in the research community for its ease of use and straightforward debugging.
    """)

# Summary of Tensorflow
col1, col2 = st.columns([1,2])
with col1:
    st.image(assets.tensorflow_logo, width=150)
with col2:
    st.subheader('TensorFlow')
    st.write("""
    TensorFlow offers both low-level APIs for detailed customization and high-level APIs like Keras for easy usability.
    - **TF 1 vs TF 2**: TensorFlow 2.x brought significant improvements over TF 1.x with eager execution by default and a more concise and understandable syntax.
    - **Scalability**: Best known for production deployments due to its robustness and comprehensive tooling.
    """)

# Summary of Keras
col1, col2 = st.columns([1,2])
with col1:
    st.image(assets.keras_logo, width=150)
with col2:
    st.subheader('Keras')
    st.write("""
    Keras is an API designed for human beings, not machines, focusing on ease of use and model prototyping.
    - **High-Level API**: Allows for fast development, making it ideal for beginners.
    - **Integration**: Works on top of TensorFlow, combining Keras' simplicity with TensorFlow's power.
    """)

# Usage Section
st.header('Usage')
st.write("""
Through the web app's pages you can compare statistics, view sections of code, try the models yourself and read my thoughts!
""")

# Footer
st.markdown("---")
st.markdown("Produced by Benjamin Hutchings | Contact me via Email or LinkedIn!")