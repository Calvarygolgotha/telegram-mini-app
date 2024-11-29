import gradio as gr
import pandas as pd
import numpy as np
from collections import defaultdict
from gradio_leaderboard import Leaderboard, SelectColumns

# Load the DataFrame from the CSV files for detailed pass@k metrics
df = pd.read_csv('results.csv')
duo_df = pd.read_csv('results_duo.csv')

# Ensure 'Model' and 'Scenario' columns are strings
df['Model'] = df['Model'].astype(str)
df['Scenario'] = df['Scenario'].astype(str)

# Function to estimate pass@k
def estimate_pass_at_k(num_samples, num_correct, k):
    def estimator(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(n, c, k) for n, c in zip(num_samples, num_correct)])

# Function to calculate pass@k
def calculate_pass_at_k(df, model, scenario, k_values=[1, 5, 10]):
    filtered_df = df[(df['Model'] == model) & (df['Scenario'] == scenario)]
    num_samples = filtered_df['Runs'].values
    num_correct = filtered_df['Successes'].values

    pass_at_k = {f"pass@{k}": estimate_pass_at_k(num_samples, num_correct, k).mean() for k in k_values}
    return pass_at_k

# Function to filter data and calculate pass@k
def filter_data(model, scenario):
    pass_at_k = calculate_pass_at_k(df, model, scenario)
    return pd.DataFrame([pass_at_k])

# Initialize the leaderboard
def init_leaderboard(dataframe, default_selection=["Model", "pass@1", "pass@5", "pass@10"], height=600):
    if dataframe is None or dataframe.empty:
        raise ValueError("Leaderboard DataFrame is empty or None.")
    return Leaderboard(
        value=dataframe,
        datatype=["markdown", "number", "number", "number"],  # Specify the types of your columns
        select_columns=SelectColumns(
            default_selection=default_selection,  # Columns to display by default
            cant_deselect=[],  # Columns that cannot be deselected
            label="Select Columns to Display:",
        ),
        search_columns=["Model"],  # Columns that can be searched
        hide_columns=[],  # Columns to hide
        filter_columns=[],  # Filters for the columns
        #bool_checkboxgroup_label="Hide models",
        interactive=False,
        height=height,
    )

# Gradio interface
models = df['Model'].unique().tolist()
scenarios = df['Scenario'].unique().tolist()

demo = gr.Blocks()

with demo:
    gr.Markdown("# 🏆 WebApp1K Models Leaderboard")
    gr.Markdown(
        "## [Discord](https://discord.gg/3qpAbWC7) " +
        "[Papers](https://huggingface.co/onekq) " +
        "[Blog](https://huggingface.co/blog/onekq/all-llms-write-great-code) "
        "[Github](https://github.com/onekq/WebApp1k) " +
        "[AI Models](https://www.aimodels.fyi/papers/arxiv/webapp1k-practical-code-generation-benchmark-web-app)")

    # Initialize leaderboard with the complete DataFrame
    duo_complete_pass_at_k = duo_df.groupby('Model')[['Runs', 'Successes']].apply(lambda x: pd.Series({
        'pass@1': estimate_pass_at_k(x['Runs'].values, x['Successes'].values, 1).mean()
    }, index=['pass@1'])).reset_index()

    complete_pass_at_k = df.groupby('Model')[['Runs', 'Successes']].apply(lambda x: pd.Series({
        'pass@1': estimate_pass_at_k(x['Runs'].values, x['Successes'].values, 1).mean(),
        'pass@5': estimate_pass_at_k(x['Runs'].values, x['Successes'].values, 5).mean(),
        'pass@10': estimate_pass_at_k(x['Runs'].values, x['Successes'].values, 10).mean()
    }, index=['pass@1', 'pass@5', 'pass@10'])).reset_index()

    gr.Markdown("# WebApp1K-Duo ([Benchmark](https://huggingface.co/datasets/onekq-ai/WebApp1K-Duo-React))")
    duo_leaderboard = init_leaderboard(duo_complete_pass_at_k, default_selection = ["Model", "pass@1"], height=400)    
    gr.Markdown("# WebApp1K ([Benchmark](https://huggingface.co/datasets/onekq-ai/WebApp1K-React))")
    leaderboard = init_leaderboard(complete_pass_at_k, height=800)

# Launch the Gradio interface
demo.launch()
