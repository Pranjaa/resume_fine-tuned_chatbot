import mlflow
import torch
import globals
from unsloth import FastLanguageModel

#Load model and tokenizer
def load_model(model_name):
  torch.cuda.empty_cache()
  print(f"Loading model: {model_name}")

  model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = globals.MAX_SEQ_LENGTH,
    load_in_4bit = True
  )

  model = FastLanguageModel.get_peft_model(
      model,
      r=globals.LORA_R,
      lora_alpha=globals.LORA_ALPHA,
      lora_dropout=globals.LORA_DROPOUT,
      bias="none",
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
      use_gradient_checkpointing="unsloth",
      random_state=3407
  )

  FastLanguageModel.for_inference(model)
  print("Model loaded and configured for inference.")

  return model, tokenizer

#Generate response based on question asked
def generate_response(question, model, tokenizer):
  instruction = f"Answer the question based on the given data. {question} Show evidence of how you reach the conclusion."
  inputs = tokenizer([globals.PROMPT_TEMPLATE.format(instruction, "", "",)], return_tensors = "pt").to("cuda")
  outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
  decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

  instruction = ""
  response = ""
  for text in decoded_outputs:
    start_index = text.find("### Instruction:")
    end_index = text.find("### Response:")

    if start_index != -1 and end_index != -1:
      instruction = text[start_index:end_index].strip()
      response = text[end_index + 13:].strip()
      break

  log_mlflow_inference(instruction, response)
  return response

#Log inference details to MLflow
def log_mlflow_inference(instruction, response):
  mlflow.log_text(instruction, "inference/instruction.txt")
  mlflow.log_text(response, "inference/response.txt")