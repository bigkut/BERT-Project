import nltk
import numpy as np
import pandas as pd
import torch
import re
import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
from nltk.corpus import stopwords


df = pd.read_csv("phishing_email.csv")
df = df.sample(n=1000, random_state=45).reset_index(drop=True)
df.dropna(inplace=True)

def convert_text(text):
    return text.lower()

df['text_combined'] = df['text_combined'].apply(convert_text)

stop_words = set(stopwords.words('english'))
phishing_words = ' '.join(df.loc[df['label'] == 1, 'text_combined']).lower().split()
filtered_words = [word for word in phishing_words if word.isalpha() and word not in stop_words]

word_freq = Counter(filtered_words).most_common(20)
"""
words, counts = zip(*word_freq)
plt.figure(figsize=(10,6))
sns.barplot(x=list(counts),y=list(words), hue=list(counts), palette='magma',legend=False)
plt.title('Top 20 words in Phishing Emails')
plt.xlabel('Frequency')
plt.show() 
"""

# split data for 20% test data
train_val_texts, test_texts, train_val_labels, text_labels = train_test_split(df['text_combined'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label'])
train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=0.125,random_state=42, stratify=train_val_labels)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#tokenize the datasets
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
# maintaining same length across emails, add 0's after 500 tokens
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

"""
print(train_encodings['input_ids'][0])
# what to pay attention to
print(train_encodings['attention_mask'][0])
"""

# Turning input ids, attention mask, and labels in a form of tensors
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)
# Preparing the dataset
train_dataset = EmailDataset(train_encodings, train_labels)
val_dataset = EmailDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

output_path = './results'
if not os.path.exists(output_path):
    os.makedirs(output_path)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()
