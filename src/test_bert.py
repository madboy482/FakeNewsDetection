from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Label index to name mapping
label_map = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true"
}

# Load model & tokenizer
model_dir = "models/bert_fake_news_final"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Input examples
texts = [
    "Breaking: The president signs new environmental law.",
    "Aliens have been spotted landing in Canada.",
    "COVID-19 vaccine increases 5G signal.",
]

# Predict each
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        print(f"{text}\n→ Class: {predicted_class} ({label_map[predicted_class]})\n")
