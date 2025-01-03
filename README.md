# NLP Assignment - Speaker Ideology & Party Status Classification

This project aims to classify parliamentary speeches based on the ideology and status of the speaker's party. The two primary tasks are:

## Tasks Overview

### Task 1: Ideology Classification
Given a parliamentary speech in one of several languages, the task is to determine the ideological leaning of the speaker's party. The classification will be binary:
- **Left (0)**: The speaker's party leans left.
- **Right (1)**: The speaker's party leans right.

### Task 2: Party Status Classification
Given a parliamentary speech in one of several languages, the task is to identify whether the speaker’s party is currently:
- **Governing (0)**: The party is in power.
- **Opposition (1)**: The party is in the opposition.

## Dataset and Country Selection
For this assignment, we have used parliamentary debates from **Turkey** to complete both tasks. 
- **Task 1**: The **English translations** of the Turkish speeches were used to classify the ideological leaning of the speaker's party (left or right).
- **Task 2**: Turkish texts were used to classify whether the speaker's party is governing or in opposition.

The data consists of Turkish speeches alongside their English translations and party affiliations.

## Code Overview

1. **Data Preprocessing and Tokenization**:
   - The dataset is loaded and preprocessed to extract the relevant columns (speech text and party labels).
   - Tokenization is performed using a multilingual BERT tokenizer (`bert-base-multilingual-cased`).
   - The dataset is split into training and validation sets, with stratification based on the party labels to ensure balanced class distribution.

2. **Task 1: Ideology Classification**:
   - The BERT-based model is used for binary classification to determine whether the speaker's party leans left or right.
   - A custom dataset class is defined for handling the tokenized inputs and labels.
   - Training is done using the Hugging Face Trainer API with appropriate configurations for learning rate, batch size, and number of epochs.

3. **Task 2: Party Status Classification**:
   - A similar process is followed for Task 2, where the model is fine-tuned for identifying whether the party is in power or in opposition.
   - The training and evaluation setup is similar to Task 1, but this task involves a different label (governing vs. opposition).

4. **GPT-2 for Additional Inference**:
   - A GPT-2 model is used for generating responses based on the parliamentary speech. The responses are then mapped to the binary labels of party status (governing or opposition).
   - The model is following a similar approach to Task 2, where responses are analyzed for the governing/opposition status of the party.

5. **Evaluation**:
   - The models are evaluated on the validation dataset, and accuracy scores are computed for both tasks. Predictions are also saved for further analysis and comparison.
