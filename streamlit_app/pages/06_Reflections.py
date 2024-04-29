import streamlit as st

from assets.asset_manager import *

assets = AssetManager()

st.header("Reflections...")
st.write(
    """
         
Despite delivering professional work in all three frameworks, this mini-project gave me an opportunity compare the frameworks side-by-side - as opposed to using them ad-hoc. 

PyTorch was like having a workshop full of tools—great for customisation, though I can imagine how it may quickly become overwhelming.

Keras was the smoothest of the trio, it’s straightforward, and gets you from point A to B quickly. Perfect when you want to get up and running without needing to get bogged down in details.

TensorFlow, without Keras, struck a middle ground, offering flexibility but also more complexity than Keras. It was a good balance, and down the line, would be more suitable for scalable end-to-end deployment and MLOPs.

Finally, the bulk of my time was spent building the Streamlit App! It's intuative for displaying analytics, comparing code and deploying demos for the models. It strikes a 
balance between a lightweight solution like a Jupyter Notebook which may be overkill in terms of code and lacks flexibility for non-technical stakeholders, and a heavier frontend stack like HTML/TailwindCSS/JavaScript/React
which would provide great scalability but takes far longer to get off the ground.

If I was to expand this project I would consider:
- More advanced architectures e.g. Convolutional Networks, Tranformers or Generative AI models (like GANs or VAEs).
- A dataset that required wrangling, cleaning and more careful pre-processing - although this is more the domain of Pandas and SciKit-Learn!
- Custom training parameters e.g. variable learning rates, saving model states etc.
- Apply user-driven model tuning, letting users adjust hyperparameters such as learning rate, epochs, batch size etc. in the App and having the models train live!

All in, this mini-project wasn’t so much of a challenge, as it was a nice excuse to get back to basics and consolidate some old skills - 
it was fun to get models trained and deployed so fast and to capture some of the nuace between the frameworks.

If you've made it this far, I hope you enjoyed the project!

Please feel free to check out more or contact me!

Best Wishes,

Ben :)

"""
)

st.image("https://media2.giphy.com/media/JLtQeoVXD5yKI/200w.gif?cid=6c09b9525oa4ptgv2ujytf0nkh50abr5ob2e6gdoeceoa728&ep=v1_gifs_search&rid=200w.gif&ct=g", caption="Happy AI-ing!")

# Footer
assets.add_footer()
