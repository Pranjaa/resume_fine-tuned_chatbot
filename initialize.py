import globals
from google.colab import drive, userdata
from inference import load_model
import mlflow

def initialize():
  global model_1, tokenizer_1, model_2, tokenizer_2
  drive.mount('/content/drive')

  setup()

  model_1, tokenizer_1 = load_model(globals.BASE_MODEL_DATASET)
  model_2, tokenizer_2 = load_model(globals.BASE_MODEL_TRAINING)

def setup():
  huggingface_setup()
  mlflow_setup()

def mlflow_setup():
  print("Setting up MLflow...")
  databricks_host = "https://community.cloud.databricks.com/"

  mlflow.login()
  mlflow.set_tracking_uri("databricks")
  mlflow.set_experiment("/check-fine-tuning-2")
  print("MLflow setup complete.")

def huggingface_setup():
  global hf_token, hf_token2
  print("Setting up HuggingFace...")
  hf_token = userdata.get('Token1')
  hf_token2 = userdata.get('Token2')
  print("HuggingFace setup complete.")

