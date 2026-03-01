import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer

model_path = "./amharic_chatbot_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(user_input):
    prompt = f"""### Instruction:
Respond naturally in Amharic.

### User:
{user_input}

### Assistant:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

while True:
    user_text = input("User: ")
    if user_text.lower() == "exit":
        break
    reply = generate_response(user_text)
    print("Assistant:", reply)