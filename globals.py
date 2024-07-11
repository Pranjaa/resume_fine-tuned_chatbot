BASE_MODEL_DATASET = "" #Your huggingface repo that contains fine-tuned model
BASE_MODEL_TRAINING = "unsloth/gemma-2b-bnb-4bit" 
NEW_MODEL = "fine-tuned-model-mistral"
DATASET_NAME = "" #Your huggingface dataset repo
DIRECTORY_PATH = "training_data"
DATA_FILE_PATH = "training_data/data.json"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
MAX_SEQ_LENGTH = 2048
OPTIMIZER = "adamw_8bit"

SOURCE_FOLDER = '/content/drive/My Drive/source_folder'
TARGET_FOLDER = '/content/drive/My Drive/target_folder'
STORE_FOLDER = '/content/drive/My Drive/store_folder'

#Template for generating prompts
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

#Template questions for generating QA pairs
TEMPLATE_QUESTIONS = {
    "What is the professional summary of {name}?",
    "Describe the key highlights of {name}'s professional background.",
    "What are the primary areas of expertise mentioned in {name}'s professional summary?",
    "What are the educational qualifications of {name}?",
    "How many years of total working experience does {name} have?",
    "How many years of working experience does {name} have at Predikly Technologies Pvt Ltd?",
    "When did {name} join Predikly Technologies Pvt Ltd?",
    "What is the role of {name} and at Predikly?",
    "List the organizations where {name} has worked.",
    "What technical skills does {name} possess?",
    "List the programming languages {name} is proficient in.",
    "Does {name} have RPA experience?",
    "List the major projects {name} has worked on.",
    "What were the roles and responsibilities of {name} in their projects?",
    "List the trainings and certifications completed by {name}."
}

#List of 4-bit models provided by unsloth
FOURBIT_MODELS = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
]