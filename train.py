import json
import torch
from datasets import Dataset
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# -------------------------
# Load Data
# -------------------------
with open("data/tiny_dialogue.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# -------------------------
# Load Model + Tokenizer
# -------------------------
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()
# model.to(device)
model.to(device)

# -------------------------
# Formatting Function
# -------------------------
def format_example(example):
    prompt = f"""### Instruction:
{example['instruction']}

### User:
{example['input']}

### Assistant:
"""
    model_input = tokenizer(
        prompt,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        example["output"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_input["labels"] = labels["input_ids"]
    return model_input

tokenized_dataset = dataset.map(format_example)

# -------------------------
# Training Arguments
# -------------------------
training_args = TrainingArguments(
    output_dir="./amharic_chatbot_model",
    per_device_train_batch_size=1,      # reduce memory
    gradient_accumulation_steps=4,      # simulate batch size 4
    num_train_epochs=10,                # enough for tiny dataset
    logging_steps=1,
    save_steps=20,
    fp16=False,                         # disable mixed precision on Windows
    learning_rate=5e-5,
    save_total_limit=2,
    report_to="none"
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# train
trainer.train()

# If your training was interrupted and you want to resume, you can use the below command
# ---- make sure you replace "checkpoint-1060" with the latest checkpoint you get on the "amharic_chatbot_model" folder
# trainer.train(resume_from_checkpoint="./amharic_chatbot_model/checkpoint-1060")


# Save final model
model.save_pretrained("./amharic_chatbot_model")
tokenizer.save_pretrained("./amharic_chatbot_model")