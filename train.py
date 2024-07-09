import globals
import mlflow
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
import os
from initialize import hf_token, hf_token2
from process_resume import fine_tuning_dataset

def train_model(model, tokenizer):
  trainer = setup_training(model ,tokenizer, fine_tuning_dataset, hf_token)

  print("Starting training.")
  trainer.train()
  print("Training complete.")

  #mlflow.log_metrics(trainer_stats.metrics)
  mlflow.log_metrics(trainer.state.log_history[-1])

def setup_training(model, tokenizer, dataset, hf_token):
  global training_arguments
  print("Setting up training arguments.")
  training_arguments=TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    optim=globals.OPTIMIZER,
    save_total_limit=3,
    save_steps=1000,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    max_grad_norm=1.0,
    max_steps=20,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=3407,
    hub_token=hf_token,
    report_to="mlflow",
    run_name="fine_tuning_job"
  )

  print("Initializing SFT Trainer.")
  trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=globals.MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=training_arguments
  )

  os.environ['OMP_NUM_THREADS'] = '1'
  return trainer

def save_trained_model(model, tokenizer):
  model.save_pretrained(globals.NEW_MODEL)
  model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",)
  model.push_to_hub_merged(f"{globals.NEW_MODEL}-merged", tokenizer, save_method = "merged_16bit", token = hf_token2)
  model.push_to_hub(globals.NEW_MODEL, tokenizer, save_method = "lora", token = hf_token2)
  mlflow.log_artifact(globals.NEW_MODEL, artifact_path="model")