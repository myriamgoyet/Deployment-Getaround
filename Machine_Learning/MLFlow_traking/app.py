# app.py
import os
import gradio as gr

def show_mlflow_url():
    return f"MLflow tracking server is available at:\n https://{os.environ['HF_SPACE_ID']}.hf.space/api/2.0/preview/mlflow"

demo = gr.Interface(fn=show_mlflow_url, inputs=[], outputs="text")
demo.launch()
