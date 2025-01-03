from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
import json

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
trainer.train(resume_from_checkpoint=True)
trainer.train()

# Evaluate the model on the validation set
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Prepare test data (if available)
test_data = pd.read_csv('val_data.tsv', sep='\t')
test_encodings = tokenizer(list(test_data['text_en']), truncation=True, padding=True, max_length=512)

test_dataset = ParliamentDataset(test_encodings, list(test_data['label']))

# Make predictions
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Save predictions
test_data['predicted_label'] = predicted_labels
test_data.to_csv('test_predictions.tsv', sep='\t', index=False)

#file_path = 'c:/Users/yhyal/OneDrive/Masaüstü/nlp2/try/test_predictions.csv'
trained_model_df = data

true_labels = trained_model_df['label']

# Extract predictions from your trained model
trained_model_predictions = trained_model_df['predicted_label'].tolist()

# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    'text': trained_model_df['text_en'],
    'true_labels': true_labels,
    'trained_model_prediction': trained_model_predictions
})

# Print the first few rows of the comparison DataFrame
print(comparison_df.head())

accuracy_trained_model = accuracy_score(comparison_df['true_labels'], comparison_df['trained_model_prediction'])

print(f"Accuracy of trained model: {accuracy_trained_model:.4f}")

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # Using the Hugging Face GPT-2 model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad_token to be the same as the eos_token (GPT-2 does not have a pad_token by default)
if tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Set environment variable to prevent memory fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load the GPT-2 model
clm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Mixed precision for GPU
    device_map="auto"  # Automatically map model layers across devices
)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clm_model.to(device)

# Prepare prompts for Task 1
task1_prompts = [
    f"Determine whether the speaker's party is left-leaning or right-leaning based on the following parliamentary speech: {text}"
    for text in val_data['text_en']
]

decoded_responses = []
max_input_length = 1024 - 20  # Reserve space for generated tokens

# Function to map GPT-2 response to binary labels (0 or 1)
def map_to_class(response):
    # Look for keywords in the response and map accordingly
    if "left" in response.lower():
        return 0
    elif "right" in response.lower():
        return 1
    return -1  # Return -1 for unclear cases (though you might want to handle this differently)

# Generate responses and post-process for comparison
for prompt in task1_prompts:
    # Tokenize with truncation
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding="longest", 
        truncation=True, 
        max_length=max_input_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU

    # Generate outputs
    outputs = clm_model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=20  # Adjust to fit within model limits
    )

    # Decode response
    decoded_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded_responses.append(decoded_response)

    # Print input and output
    print(f"Input: {prompt}\nOutput: {decoded_response}\n")

# Save the GPT-2 predictions to a JSON file
with open("task1_clm_predictions.json", "w") as f:
    json.dump(decoded_responses, f)

# Function to map GPT-2 response to binary labels (0 or 1)
def map_to_class(response):
    # Look for keywords in the response and map accordingly
    if "left" in response.lower():
        return 0
    elif "right" in response.lower():
        return 1
    return -1  # Return -1 for unclear cases (though you might want to handle this differently)
# Load the GPT-2 predictions from the saved JSON file
with open("task1_clm_predictions.json", "r") as f:
    gpt2_predictions = json.load(f)

# Post-process the GPT-2 predictions to binary labels
gpt2_predictions_mapped = [map_to_class(response) for response in gpt2_predictions]

true_labels = val_data['label']

# Calculate the accuracy of GPT-2 model
accuracy_gpt2 = accuracy_score(true_labels, gpt2_predictions_mapped)

# Print the accuracy
print(f"Accuracy of GPT-2 model: {accuracy_gpt2}")

# Load GPT-2 model and tokenizer
model_name = "ytu-ce-cosmos/turkish-gpt2-large"  # Using the Hugging Face GPT-2 model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad_token to be the same as the eos_token (GPT-2 does not have a pad_token by default)
if tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Set environment variable to prevent memory fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load the GPT-2 model
clm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Mixed precision for GPU
    device_map="auto"  # Automatically map model layers across devices
)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clm_model.to(device)

# Prepare prompts for Task 1
task1_prompts_turkish = [
    f"Aşağıdaki meclis konuşmasına dayanarak, konuşmacının partisinin sol eğilimli mi yoksa sağ eğilimli mi olduğunu belirleyin: {text}"
    for text in val_data['text']
]

decoded_responses_turkish = []
max_input_length = 1024 - 20  # Reserve space for generated tokens

# Function to map GPT-2 response to binary labels (0 or 1)
def map_to_class(response):
    # Look for keywords in the response and map accordingly
    if "sol" in response.lower():
        return 1
    elif "sağ" in response.lower():
        return 0
    return -1  # Return -1 for unclear cases (though you might want to handle this differently)

# Generate responses and post-process for comparison
for prompt in task1_prompts_turkish:
    # Tokenize with truncation
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding="longest", 
        truncation=True, 
        max_length=max_input_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU

    # Generate outputs
    outputs = clm_model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=20  # Adjust to fit within model limits
    )

    # Decode response
    decoded_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded_responses_turkish.append(decoded_response)

    # Print input and output
    print(f"Input: {prompt}\nOutput: {decoded_response}\n")

# Save the GPT-2 predictions to a JSON file
with open("task1_clm_predictionsTR.json", "w") as f:
    json.dump(decoded_responses_turkish, f)

# Post-process the GPT-2 predictions to binary labels
gpt2_predictions_mapped_turkish = [map_to_class(response) for response in decoded_responses_turkish]

true_labels_turkish = val_data['label']

# Calculate the accuracy of GPT-2 model for Turkish
accuracy_gpt2_turkish = accuracy_score(true_labels_turkish, gpt2_predictions_mapped_turkish)

# Print the accuracy
print(f"Accuracy of GPT-2 model on Turkish data: {accuracy_gpt2_turkish}")
