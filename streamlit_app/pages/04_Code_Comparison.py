import streamlit as st
import difflib

st.title('Code Comparison')

# Sample code snippets
code_pytorch = """def train():
    # PyTorch training loop
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()"""

code_tensorflow = """def train():
    # TensorFlow training loop
    for epoch in range(epochs):
        for images, labels in dataset:
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_fn(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))"""

# Display the code
st.write("## PyTorch Training Code")
st.code(code_pytorch, language='python')

st.write("## TensorFlow Training Code")
st.code(code_tensorflow, language='python')

# Compare the code
st.write("## Code Differences")
diff = difflib.HtmlDiff().make_file(code_pytorch.splitlines(), code_tensorflow.splitlines())
st.components.v1.html(diff, height=400, scrolling=True)
