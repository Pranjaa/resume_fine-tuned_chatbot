import globals
from mlflow_log import log_mlflow_inference

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