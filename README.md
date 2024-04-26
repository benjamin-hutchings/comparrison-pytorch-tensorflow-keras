This needs tidying up!!

## Project Overview

This project demonstrates how to build the same deep learning model across three different frameworks: PyTorch, Keras, and TensorFlow. The objective is to illustrate the differences in syntax and structure when using these frameworks and to compare their performance on a common task - in this case, producing a simple feedforward network trained to classify images in the MINST handwriting dataset.

The scripts are availiable in juptyer notebooks and are hosted in a Streamlit web app.

The Streamlit App (ADD LINK HERE) provides an comprehensive overview, code comparison, quantitative analysis and to try the models for yourself!

# How to Use

Explore the Streamlit app to compare how each framework handles:

Data loading and preprocessing
Model definition and construction
Training loops
Evaluation and inference
The Jupyter notebooks provide detailed code annotations and explanations for each step in the model lifecycle.

## Requirements

- Python 3.8+
- PyTorch
- TensorFlow
- Keras
- Streamlit

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/benjamin-hutchings/comparrison-pytorch-tensorflow-keras.git
cd comparrison-pytorch-tensorflow-keras
```

## Host the Streamlit Web App locally

Run the bash file:

...

Or manually:

```bash
pip install streamlit
cd streamlit_app
streamlit run .\00_DL_Framework_Comparrison.py
```

# Setting Up a Virtual Environment

It's recommended to use a virtual environment to manage the dependencies for your project. If you're using venv or conda, here's how you can set up a virtual environment for this project:

For venv:

'''bash
python -m venv venv
# Activate the environment
# On Windows
venv\Scripts\activate
# On MacOS/Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt


For Conda:
'''bash
Copy code
conda create --name dl-framework-comparison python=3.8
conda activate dl-framework-comparison

# Install PyTorch (visit PyTorch's installation page to customize the command for your system)
conda install pytorch torchvision -c pytorch

# Install TensorFlow and Keras
conda install tensorflow keras

# Install Streamlit
pip install streamlit
'''

