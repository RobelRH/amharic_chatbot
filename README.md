# Amharic Conversational Chatbot

A simple **Amharic conversational chatbot** built using the `mT5` model from Hugging Face Transformers. This chatbot is trained on a small custom dataset and demonstrates how to fine-tune a multilingual model for Amharic dialogue generation.

---

## 🚀 Features

- Fine-tunes `google/mt5-small` for Amharic conversations.
- Uses a JSON-based dataset for easy expansion.
- Supports GPU acceleration if available.
- Ready for extension with larger datasets or more advanced training.

---

## 🛠️ Requirements

- Python 3.9+
- PyTorch 2.x
- Transformers
- Datasets
- Optional: GPU with CUDA for faster training

Install via pip:

```bash
pip install torch transformers datasets

Or using Miniconda:

conda create -n amharic_chatbot python=3.9
conda activate amharic_chatbot
pip install torch transformers datasets
📂 Project Structure
amharic_chatbot/
├─ data/
│  └─ tiny_dialogue.json        # Training dataset
├─ amharic_chatbot_model/       # Directory where model is saved
├─ train.py                     # Training script
└─ README.md

📝 Dataset Format

Dataset should be JSON with entries like:

{
    "instruction": "Respond naturally in Amharic.",
    "input": "ሰላም",
    "output": "ሰላም! እንዴት ነህ?"
}

instruction: Optional instruction for the model.

input: User message.

output: Expected chatbot response.

⚙️ Training

Run the training script:

python train.py

Training details:

Model: google/mt5-small

Batch size: 1 (gradient accumulation simulates batch size 4)

Epochs: 10

Learning rate: 5e-5

Mixed precision: Disabled (fp16=False) on Windows

💾 Saving & Resuming

Model and tokenizer are saved in ./amharic_chatbot_model.

Resume training from checkpoint:

trainer.train(resume_from_checkpoint="./amharic_chatbot_model/checkpoint-1060")

Replace checkpoint-1060 with the latest checkpoint.

🔧 Usage After Training
from transformers import AutoTokenizer, MT5ForConditionalGeneration
import torch

model_dir = "./amharic_chatbot_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = MT5ForConditionalGeneration.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_text = "ሰላም፣ እንዴት ነህ?"
prompt = f"### Instruction:\nRespond naturally in Amharic.\n\n### User:\n{input_text}\n\n### Assistant:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=128)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
🔮 Future Improvements

Expand dataset for better conversational quality.

Add an interactive chatbot interface (CLI, web, or Telegram bot).

Experiment with larger multilingual models (mt5-base, mt5-large).

Add evaluation metrics for dialogue generation.

📄 License

This project is open-source and available under the MIT License.