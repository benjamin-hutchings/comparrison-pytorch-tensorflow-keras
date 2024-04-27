from assets.asset_manager import *
import streamlit as st
import pandas as pd
import plotly.express as px

assets = AssetManager()

# Page Title
st.title("Statistical Comparisons")

# Introduction
st.markdown(
    """
This interactive dashboard presents a comparison of cumulative runtime and the lines of code required across various stages of model development
for the three popular deep learning frameworks and their respective training and evaluation metrics.

Use the navigator on the left to view different metrics, and the widgets to interact with the graphs

"""
)


# Load CSV data
@st.cache_resource
def load_data(datapath):
    return pd.read_csv(datapath)


sections = ["Runtime", "Usability", "Performance"]

selected_section = st.sidebar.selectbox("Choose a topic", sections)

if selected_section == "Runtime":
    st.write("## Runtime")
    st.write(
        """
             Take a look at the cumulative runtime for sections of model development and inferences made.
             """
    )
    # Runtime
    runtime_data = load_data("streamlit_app/assets/script_data/runtime.csv")

    # Convert all non-framework columns to cumulative sums
    cumulative_data = runtime_data.set_index("Framework").cumsum(axis=1).reset_index()

    # Slider for selecting the number of stages
    max_stages = (
        len(cumulative_data.columns) - 1
    )  # subtracting 1 for the 'Framework' column
    selected_stages = st.slider(
        "Slide to select the number of stages to display", 3, max_stages, max_stages
    )

    # Adjust the dataframe based on the slider
    cumulative_data_selected = cumulative_data.iloc[
        :, : selected_stages + 1
    ]  # +1 to include the 'Framework' column

    # Preparing data for plotting
    # We need to melt the adjusted dataframe to work nicely with Plotly
    melted_data = cumulative_data_selected.melt(
        id_vars=["Framework"], var_name="Stage", value_name="Cumulative Time"
    )

    # Plotting using Plotly Express
    fig = px.line(
        melted_data,
        x="Stage",
        y="Cumulative Time",
        color="Framework",
        markers=True,
        title="Cumulative Runtime across Selected Framework Stages",
    )

    # Enhance the layout
    fig.update_layout(
        xaxis_title="Stages",
        yaxis_title="Cumulative Time (seconds)",
        legend_title="Framework",
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Insights Section
    st.write("### Insights and Analysis")
    st.markdown(
        """
    - **Initial Setup Time:** TensorFlow appears to have a significantly longer total time during the initial setup 
    stages (installation, imports, and loading), which could impact developer productivity for smaller projects or 
    experiments.
    - **Training Time:** PyTorch and TensorFlow show substantial times in the training stage, with TensorFlow 
    taking the longest. This could be indicative of more robust handling of computation but may also suggest a 
    higher computational overhead.
    - **Evaluation and Visualization:** All frameworks perform relatively quickly in the final stages of visualization 
    and evaluation, indicating that these stages are less computationally intensive or are optimized across all three 
    frameworks.
    
    It is important to note that this experiement was ran on a CPU, if alternative hardware was targetting the results may vary.
    """
    )

elif selected_section == "Usability":
    st.write("## Lines of Code")
    st.write(
        """
        One rudimentary metric of usability is the required lines of code per stage - this doesn't give a full picture
        but provides some indication of complexity.
        
        Select stages to view:
        """
    )

    # Load data
    code_lines_data = load_data("streamlit_app/assets/script_data/code_lines.csv")

    # Get unique stages from the DataFrame
    stages = code_lines_data["Stage"].unique()

    # Organizing checkboxes in a more compact layout using columns
    col1, col2, col3 = st.columns(3)  # Adjust the number of columns based on preference
    with col1:
        selected_stages = []
        for i, stage in enumerate(stages):
            if i % 3 == 0:  # Adjust the modulo operation based on the number of columns
                if st.checkbox(stage, True, key=stage):
                    selected_stages.append(stage)
    with col2:
        for i, stage in enumerate(stages):
            if i % 3 == 1:
                if st.checkbox(stage, True, key=stage):
                    selected_stages.append(stage)
    with col3:
        for i, stage in enumerate(stages):
            if i % 3 == 2:
                if st.checkbox(stage, True, key=stage):
                    selected_stages.append(stage)

    # Filter data based on selected stages
    filtered_data = code_lines_data[code_lines_data["Stage"].isin(selected_stages)]

    # Creating a bar chart using Plotly Express
    if not filtered_data.empty:
        fig = px.bar(
            filtered_data,
            x="Stage",
            y="Lines",
            color="Framework",
            barmode="group",
            title="Lines of Code per Development Stage by Framework",
            labels={"Lines": "Lines of Code", "Stage": "Development Stage"},
        )

        # Update layout
        fig.update_layout(
            xaxis_title="Stage", yaxis_title="Lines of Code", legend_title="Framework"
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
    else:
        st.write("Please select at least one stage to display.")

    st.write("### Insights and Analysis")
    st.write(
        """
        - **Framework Comparison:** TensorFlow tends to require more lines of code particularly in the training and initial setup stages, suggesting a possibly higher complexity or more robust functionality.
        - **Training Stage:** Training stage sees the highest variation in lines of code, with TensorFlow showing the most complexity, followed by Pytorch and then Keras, which has significantly fewer lines of code. This might indicate ease of use in simpler projects or faster prototyping capabilities with Keras.
        - **Visualization Consistency:** All three frameworks show similar lines of code for the visualization stage, suggesting a standardization in visualization libraries or methodologies.
        - **Initialization and Imports:** Pytorch and TensorFlow show more complexity in initial stages compared to Keras, which may imply a steeper learning curve but potentially more control or customization options.
        
        It's essential to consider these factors when choosing a framework, especially based on project requirements and team expertise.
        """
    )

elif selected_section == "Performance":
    st.write("## Model Performance")
    st.write(
        """
        Understanding the performance of different frameworks through their training and test accuracy provides a more concrete measure of model effectiveness. Here, we also visualize the training, validation, and test accuracies for each framework.
        """
    )

    # Load performance data
    performance_data = load_data("streamlit_app/assets/script_data/performance.csv")

    # Display model performance metrics
    st.write("### Training, Validation, and Test Accuracies")

    # Create a grouped bar chart
    fig = px.bar(performance_data, x='Framework', y='Accuracy', color='TVT', barmode='group',
                 title="Accuracy Comparison across Training, Validation, and Test",
                 labels={"Accuracy": "Accuracy (%)", "Framework": "Framework", "TVT": "Type"})
    
    # Update layout
    fig.update_layout(xaxis_title="Framework",
                      yaxis_title="Accuracy (%)",
                      legend_title="Evaluation Type")
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)
    
    # Display training statistics images
    st.write('''
             ### Training Statistics Visualizations
             Graphs of training and validation accuracy, and loss, vs epochs - produced with matplotlib.
             ''')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("./assets/imgs/py_output.png", caption="Pytorch Training Statistics")
    with col2:
        st.image("./assets/imgs/tf_output.png", caption="TensorFlow Training Statistics")
    with col3:
        st.image("./assets/imgs/keras_output.png", caption="Keras Training Statistics")

    st.write("### Insights and Analysis")
    st.markdown(
        """
        - **Test Accuracy:** All frameworks demonstrate robust training capabilities with very high training and validation accuracies, but TensorFlow shows exceptional generalization as seen from its higher test accuracy relative to its validation accuracy.
        - **TensorFlow:** The model not only reached high accuracies but also maintained a steady improvement over epochs with a lower and decreasing loss, indicating effective learning and optimization.
        - **Keras:** Keras shows rapid attainment of high accuracy in initial epochs, which could be advantageous for scenarios requiring quick iterations.
        - **Pytorch :** Pytorch shows a very close performance between training, validation, and test datasets, indicating good model stability and less likelihood of overfitting compared to others.
        """
    )

# Footer
assets.add_footer()