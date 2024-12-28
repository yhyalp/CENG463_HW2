from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from accelerate import init_empty_weights, dispatch_model

print(os.getcwd(), "sssssssssssssssssssssssssssssssssssssssss")
print(torch.cuda.is_available())

# Load the dataset into a Pandas DataFrame
file_path = 'c:/Users/yhyal/OneDrive/Masaüstü/nlp2/try/orientation-tr-train.tsv'
data = pd.read_csv(file_path, sep='\t')

# Check the first few rows
print(data.head())

# Check class distribution
print("Label distribution:", Counter(data['label']))

# Perform stratified split
train_data, val_data = train_test_split(
    data,
    test_size=0.1,  # 10% for validation
    stratify=data['label'],  # Ensure label proportions are maintained
    random_state=42  # Set seed for reproducibility
)

# Check label distribution in training and validation sets
print("Train label distribution:", Counter(train_data['label']))
print("Validation label distribution:", Counter(val_data['label']))

# Analyze class distribution in the training set
label_counts = train_data['label'].value_counts()
print("Training set class distribution:\n", label_counts)

# Calculate imbalance ratio
imbalance_ratio = label_counts.min() / label_counts.max()
print("Imbalance ratio:", imbalance_ratio)

train_data.to_csv('train_data.tsv', sep='\t', index=False)
val_data.to_csv('val_data.tsv', sep='\t', index=False)

train_data = pd.read_csv('train_data.tsv', sep='\t')
val_data = pd.read_csv('val_data.tsv', sep='\t')

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
train_encodings = tokenizer(list(train_data['text_en']), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_data['text_en']), truncation=True, padding=True, max_length=512)

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

train_dataset = ParliamentDataset(train_encodings, list(train_data['label']))
val_dataset = ParliamentDataset(val_encodings, list(val_data['label']))

model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",  # Evaluate at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="steps",        # Save model at the end of each epoch
    save_steps=100,               # This will be ignored since save_strategy is "epoch"
    logging_steps=100,            # Log every 200 steps
    load_best_model_at_end=True   # Load the best model based on evaluation
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
#trainer.train(resume_from_checkpoint=True)
#trainer.train()
"""
# Evaluate the model on the validation set
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Prepare test data (if available)
test_data = pd.read_csv('val_data.tsv', sep='\t')  # Replace with your actual test dataset
test_encodings = tokenizer(list(test_data['text_en']), truncation=True, padding=True, max_length=512)

test_dataset = ParliamentDataset(test_encodings, list(test_data['label']))

# Make predictions
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Save predictions
test_data['predicted_label'] = predicted_labels
test_data.to_csv('test_predictions.tsv', sep='\t', index=False)
"""
# Load CLM
access_token = "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"
model_name = "openlm-research/open_llama_3b"  # Replace with the actual model name

# Load tokenizer with use_auth_token
tokenizer = AutoTokenizer.from_pretrained(
    "openlm-research/open_llama_3b",
    token=access_token,
    legacy=False  # Use new tokenizer behavior
)

# Set the pad_token to be the same as the eos_token (if eos_token exists)
if tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token
else:
    # Optionally, define a custom pad_token if eos_token does not exist
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Load model with use_auth_token
clm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=access_token,
    torch_dtype=torch.float16,  # Use mixed precision
    device_map="auto"  # Automatically map model layers across devices
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clm_model.to(device)
clm_model.gradient_checkpointing_enable()
# Prepare prompts for Task 1
task1_prompts = [
    f"Determine whether the speaker's party is left-leaning or right-leaning based on the following parliamentary speech: {text}"
    for text in val_data['text_en']
]

# Tokenize inputs with padding and truncation
inputs = tokenizer(task1_prompts, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move inputs to GPU if needed

# Add attention mask to ensure proper behavior
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Generate outputs
outputs = clm_model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=20)

# Decode responses
decoded_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Print inputs and outputs
for text, response in zip(task1_prompts[:2], decoded_responses):  # Matching the reduced batch size
    print(f"Input: {text}\nOutput: {response}\n")
