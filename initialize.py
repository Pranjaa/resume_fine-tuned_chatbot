import globals
from google.colab import drive, userdata
from inference import load_model
import mlflow

def initialize():
  global model_1, tokenizer_1, model_2, tokenizer_2
  drive.mount('/content/drive')

  mlflow_setup()

  model_1, tokenizer_1 = load_model(globals.BASE_MODEL_DATASET)
  model_2, tokenizer_2 = load_model(globals.BASE_MODEL_TRAINING)

  return model_1, tokenizer_1, model_2, tokenizer_2

def mlflow_setup():
  print("Setting up MLflow...")
  databricks_host = "https://community.cloud.databricks.com/"

  mlflow.login()
  mlflow.set_tracking_uri("databricks")
  mlflow.set_experiment("/check-fine-tuning-2")
  print("MLflow setup complete.")