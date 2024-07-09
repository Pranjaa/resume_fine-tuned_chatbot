import globals
from train import train_model, save_trained_model
from initialize import model_1, tokenizer_1, model_2, tokenizer_2
import json
from glob import glob
import os
import shutil
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import hf_hub_download, login, upload_file
from initialize import hf_token2
from mlflow_log import log_mlflow_params

def process_resume(files):
    upload_files_to_drive(files)
    generate_QA_pairs(model_1, tokenizer_1)
    split_data()
    setup_fine_tuning_dataset(tokenizer_2)
    train_model(model_2, tokenizer_2)
    save_trained_model(model_2, tokenizer_2)
    log_mlflow_params()

def upload_files_to_drive(files):
  os.makedirs(globals.TARGET_FOLDER, exist_ok=True)

  uploaded_files = files.upload()

  for filename in uploaded_files.keys():
    destination_path = os.path.join(globals.TARGET_FOLDER, filename)
    shutil.move(filename, destination_path)
    print(f"File {filename} uploaded to {destination_path}")

def generate_QA_pairs(model, tokenizer):
  print("Generating QA pairs.")
  all_results = []

  if os.path.exists(globals.DATA_FILE_PATH):
    with open(globals.DATA_FILE_PATH, "r") as json_file:
      all_results = json.load(json_file)

  for text_file in glob(os.path.join(globals.TARGET_FOLDER, "*.txt")):
    with open(text_file, "r", encoding="utf-8") as file:
      print(f"Reading file: {text_file}")
      context = file.read()

    context = context.replace("present", "01/07/24")
    name = os.path.basename(text_file)

    for query in globals.TEMPLATE_QUESTIONS:
      instruction = f"Answer the question based on the given data. {query.format(name=name)} Show evidence of how you reach the conclusion. Include name of person in the output."
      inputs = tokenizer([globals.PROMPT_TEMPLATE.format(instruction, context, "",)], return_tensors = "pt").to("cuda")
      outputs = model.generate(**inputs, max_new_tokens = 300, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
      decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

      response = ""
      for text in decoded_outputs:
        start_index = text.find("### Response:")
        if start_index != -1:
          response = text[start_index + 13:].strip()
          break

      result = {
        "instruction": instruction,
        "input": context,
        "output": response
      }
      all_results.append(result)

    with open(globals.DATA_FILE_PATH, "w", encoding="utf-8") as json_file:
      json.dump(all_results, json_file, indent=4)

    store_file_path = os.path.join(globals.STORE_FOLDER, os.path.basename(text_file))
    shutil.move(text_file, store_file_path)
    print(f"File moved to: {store_file_path}")

  print("QA pairs generated and saved.")

def split_data(train_ratio=0.8):
  print("Splitting data into training and validation sets...")
  filepath = "training_data/data.json"

  if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

  with open(filepath, "r", encoding="utf-8") as file:
      try:
          data = json.load(file)
      except json.JSONDecodeError as e:
          print(f"Error decoding JSON: {e}")
          return

  if not data:
      print("No valid data found in the file.")
      return

  train_size = int(len(data) * train_ratio)
  train_data = data[:train_size]
  valid_data = data[train_size:]

  train_filepath="training_data/train.json"
  valid_filepath="training_data/valid.json"

  with open(train_filepath, "w", encoding="utf-8") as file:
    for item in train_data:
      file.write(json.dumps(item) + "\n")
  with open(valid_filepath, "w", encoding="utf-8") as file:
    for item in valid_data:
      file.write(json.dumps(item) + "\n")

  train_dataset = Dataset.from_json(train_filepath)
  valid_dataset = Dataset.from_json(valid_filepath)

  dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
  })

  login(token=hf_token2, add_to_git_credential=True)
  existing_file_path = hf_hub_download(repo_id=globals.DATASET_NAME, repo_type="dataset", filename="data.json")

  with open(existing_file_path, "r", encoding="utf-8") as file:
      existing_data = json.load(file)

  combined_train_data = existing_data[:len(existing_data) // 2] + train_data
  combined_valid_data = existing_data[len(existing_data) // 2:] + valid_data

  combined_data = {
      "train": combined_train_data,
      "validation": combined_valid_data
  }

  combined_train_filepath = "combined_train.json"
  combined_valid_filepath = "combined_valid.json"

  with open(combined_train_filepath, "w", encoding="utf-8") as file:
      json.dump(combined_train_data, file, ensure_ascii=False, indent=4)
  with open(combined_valid_filepath, "w", encoding="utf-8") as file:
      json.dump(combined_valid_data, file, ensure_ascii=False, indent=4)

  upload_file(
      path_or_fileobj=combined_train_filepath,
      path_in_repo="train.json",
      repo_id="Pranja/Resumes",
      repo_type="dataset"
  )
  upload_file(
      path_or_fileobj=combined_valid_filepath,
      path_in_repo="validation.json",
      repo_id="Pranja/Resumes",
      repo_type="dataset"
  )

  print(f"Dataset updated and pushed to Hugging Face Hub: Pranja/Resumes")

def setup_fine_tuning_dataset(tokenizer):
  def format_prompt(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []

    EOS_TOKEN = tokenizer.eos_token

    for instruction, input, output in zip(instructions, inputs, outputs):
        text = globals.PROMPT_TEMPLATE.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

  global fine_tuning_dataset
  print(f"Loading fine-tuning dataset: {globals.DATASET_NAME}")

  dataset = load_dataset(globals.DATASET_NAME)
  fine_tuning_dataset = dataset.map(format_prompt, batched = True)

  print("Fine-tuning dataset loaded and formatted.") 