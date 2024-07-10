import os
import json
import shutil
import mlflow
import globals
from glob import glob
from config import hf_token2
from train import train_model, save_trained_model
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import hf_hub_download, login, upload_file

def process_resume(files, model_1, tokenizer_1, model_2, tokenizer_2):
    upload_files_to_drive(files)
    generate_QA_pairs(model_1, tokenizer_1)
    split_data()
    fine_tuning_dataset = setup_fine_tuning_dataset(tokenizer_2)
    training_arguments = train_model(model_2, tokenizer_2, fine_tuning_dataset)
    save_trained_model(model_2, tokenizer_2)
    log_mlflow_params(training_arguments)

def upload_files_to_drive(files):
    os.makedirs(globals.TARGET_FOLDER, exist_ok=True)
    os.makedirs(globals.STORE_FOLDER, exist_ok=True)

    for uploaded_file in files:
        filename = uploaded_file.name
        destination_path = os.path.join(globals.TARGET_FOLDER, filename)

        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        shutil.move(filename, destination_path)
        print(f"File {filename} uploaded to {destination_path}")

def generate_QA_pairs(model, tokenizer):
  print("Generating QA pairs.")
  all_results = []

  os.makedirs(globals.DIRECTORY_PATH, exist_ok=True)
  data_file_path = os.path.join(globals.DATA_FILE_PATH, "data.json")

  os.makedirs(globals.STORE_FOLDER, exist_ok=True)

  if os.path.exists(data_file_path):
    with open(data_file_path, "r", encoding="utf-8") as json_file:
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

    store_file_path = os.path.join(globals.STORE_FOLDER, "")
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

  return fine_tuning_dataset

def log_mlflow_params(training_arguments):
  with mlflow.start_run() as run:
    mlflow.log_params({
      "base_model": globals.BASE_MODEL_TRAINING,
      "new_model": globals.NEW_MODEL,
      "dataset_name": globals.DATASET_NAME,
      "lora_r": globals.LORA_R,
      "lora_alpha": globals.LORA_ALPHA,
      "lora_dropout": globals.LORA_DROPOUT,
      "max_seq_length": globals.MAX_SEQ_LENGTH,
      "output_dir": training_arguments.output_dir,
      "batch_size": training_arguments.per_device_train_batch_size,
      "eval_size": training_arguments.per_device_eval_batch_size,
      "gradient_accumulation_steps": training_arguments.gradient_accumulation_steps,
      "optimizer": globals.OPTIMIZER,
      "save_steps": training_arguments.save_steps,
      "logging_steps": training_arguments.logging_steps,
      "learning_rate": training_arguments.learning_rate,
      "fp16": training_arguments.fp16,
      "bf16": training_arguments.bf16,
      "gradient_checkpointing": training_arguments.gradient_checkpointing,
      "evaluation_strategy": training_arguments.evaluation_strategy,
      "save_strategy": training_arguments.save_strategy,
      "max_grad_norm": training_arguments.max_grad_norm,
      "max_steps": training_arguments.max_steps,
      "num_train_epochs": training_arguments.num_train_epochs,
      "weight_decay": training_arguments.weight_decay,
      "warmup_steps": training_arguments.warmup_steps,
      "lr_scheduler_type": training_arguments.lr_scheduler_type,
      "load_best_model_at_end": training_arguments.load_best_model_at_end,
      "metric_for_best_model": training_arguments.metric_for_best_model,
      "greater_is_better": training_arguments.greater_is_better,
      "seed": training_arguments.seed,
    })