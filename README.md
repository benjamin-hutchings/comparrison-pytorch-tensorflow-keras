# Project Overview

This project demonstrates how to build the same deep learning model in three different frameworks: PyTorch, Keras, and TensorFlow. The objective is to illustrate the differences in syntax and structure when using these frameworks and to compare their performance on a common task - in this case, producing a simple feedforward network trained to classify images in the MINST handwriting dataset.

The deep learning scripts are availiable in Juptyer Notebooks.

# Streamlit

The project is in a Streamlit web app, availiable in a live demo here:

https://comparrison-pytorch-tensorflow-keras-kfvns3i3ex4rdxbtcjpwk4.streamlit.app/

The App provides an comprehensive overview, code comparison, interactive quantitative analysis and a demo for you to try the models with any image!

# How to Use

Explore the Streamlit app to compare how each framework handles:

- Data loading and preprocessing
- Model definition and construction
- Training loops
- Evaluation and inference
- Overall metrics for complexity and accuracy

The Jupyter notebooks provide detailed code annotations and explanations for each step in the model lifecycle.

## Requirements

- Python 3.8+
- PyTorch
- TensorFlow
- Keras
- Streamlit
- Pandas

No GPU targetting is required, all models were trained on CPU.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/benjamin-hutchings/comparrison-pytorch-tensorflow-keras.git
cd comparrison-pytorch-tensorflow-keras
```

## Host the Streamlit Web App locally

```bash
pip install streamlit
cd streamlit_app
streamlit run .\00_DL_Framework_Comparrison.py
```

Please note you will have to change the file locations to relative paths (add ../) for the .csv files and saved models for full functionality if hosted locally.

## Thanks for viewing!

Any questions or feedback please reachout to me via my Website or LinkedIn.

Best,

Ben :)


