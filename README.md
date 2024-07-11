# Resume Chatbot

A chatbot that answers questions based on input resumes. The main script is run on Google Colab, with setup and functioning instructions provided below.

## Setup HuggingFace

1. Create Account: Go to `huggingface.co`, create an account and log in.
2. Get Access Tokens:
    - On the top right, go to settings.
    - Click on "Access Tokens".
    - Create two tokens: 
        - 'hf_token' with type 'fine-grained (custom)'
        - 'hf_token2' with type 'write'
3. Clone Github Repo: Add a 'config.py' file containing 'hf_token' and 'hf_token2' to the folder.
4. Model Repo: Create a variable 'NEW_MODEL' with the name of the desired model and save it in 'globals.py'.
5. Dataset Repo: Create a dataset repository. Copy the path and create a variable called 'DATASET_NAME' in 'globals.py'.

## Setup MLflow

1. This is the free hosted tracking server method (Databricks Community Edition).
2. Create Account: Go to `https://community.cloud.databricks.com/` and create a free account.
3. Install dependencies: 
    ```
    %pip install -q mlflow databricks-sdk
    ```
4. Setup Databricks CE Authentication:
    - Databricks host: `https://community.cloud.databricks.com/`
    - Username: Your Databricks CE email address
    - Password: Your Databricks CE password
5. View Experiment: Log in to Databricks CE and clicks on 'Experiements' on the left sidebar.

## Setup Google Drive

1. In your Google Drive, create folders to store dataset.
2. Ensure all permissions are granted to Colab when the drive is mounted.

## Setup Ngrok

1. Create Account: Go to `ngrok.com`, create an account, and log in.
2. Get Auth Token: 
    - On the left sidebar, click on "AuthTokens" under "Tunnels".
    - Click on "Add Tunnel Authtoken" with default values and save the "auth_token".

## Execution

The main scrpt runs on Google Colab utilizing the free T4 GPU offered. 

