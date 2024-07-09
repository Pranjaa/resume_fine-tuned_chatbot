import globals
import mlflow
from train import training_arguments

def log_mlflow_params():
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

def log_mlflow_inference(instruction, response):
  mlflow.log_text(instruction, "inference/instruction.txt")
  mlflow.log_text(response, "inference/response.txt")