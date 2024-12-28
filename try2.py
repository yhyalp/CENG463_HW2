import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM

# Load the dataset
file_path = 'c:/Users/yhyal/OneDrive/Masaüstü/nlp2/try/power-tr-train.tsv'
data = pd.read_csv(file_path, sep='\t')

# Task 2: Classify whether the party is governing (0) or in opposition (1)
task2_data = data[['text', 'gov_opposition_label']].rename(columns={'gov_opposition_label': 'label'})

# Stratified split into training and validation sets
train_data, val_data = train_test_split(
    task2_data,
    test_size=0.1,  # 10% for validation
    stratify=task2_data['label'],  # Maintain label proportions
    random_state=42
)

# Tokenize the data (use Turkish text)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
train_encodings = tokenizer(list(train_data['text']), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_data['text']), truncation=True, padding=True, max_length=512)

class ParliamentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create datasets
train_dataset = ParliamentDataset(train_encodings, list(train_data['label']))
val_dataset = ParliamentDataset(val_encodings, list(val_data['label']))

# Fine-tune multilingual BERT
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results_task2',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs_task2',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./task2_finetuned_model')
tokenizer.save_pretrained('./task2_finetuned_model')

# Evaluate the fine-tuned model
trainer.evaluate()

"""
# Zero-shot inference using Llama-3.1-8B (causal language model)
causal_model_name = "meta-llama/Llama-3.1-8B"
causal_model = AutoModelForCausalLM.from_pretrained(causal_model_name, device_map="auto", torch_dtype=torch.float16)

# Tokenizer for causal language model
causal_tokenizer = AutoTokenizer.from_pretrained(causal_model_name)

# Perform inference on Turkish text
test_text_turkish = val_data['text'].iloc[0]  # Example test text in Turkish
inputs_turkish = causal_tokenizer(test_text_turkish, return_tensors="pt").to('cuda')

# Generate predictions
causal_output_turkish = causal_model.generate(**inputs_turkish, max_new_tokens=50)
print("Turkish Text Inference:", causal_tokenizer.decode(causal_output_turkish[0], skip_special_tokens=True))

# Perform inference on English text
test_text_english = val_data['text_en'].iloc[0]  # Example test text in English
inputs_english = causal_tokenizer(test_text_english, return_tensors="pt").to('cuda')

# Generate predictions
causal_output_english = causal_model.generate(**inputs_english, max_new_tokens=50)
print("English Text Inference:", causal_tokenizer.decode(causal_output_english[0], skip_special_tokens=True))
"""