# -*- coding: utf-8 -*-
"""““data_preprocessing.ipynb””

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bODIDwv_FV2wHbIB4igN60xuyOBNSjLY
"""


import nltk
nltk.download('punkt_tab')

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
import torch
from sklearn.preprocessing import LabelEncoder

import numpy as np


# Convert to PyTorch datasets
class MBTIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    

def main() :

    # Loading the extracted CSV file
    csv_path = 'mbti_data.csv'
    #csv_path = 'mbti_1.csv'
    df = pd.read_csv(csv_path)

    # Displaying the first few rows to understand its structure
    df.head()

    """Split the MBTI type to subcategory"""

    df['I/E'] = df['type'].apply(lambda x: 'I' if 'I' in x else 'E')
    df['N/S'] = df['type'].apply(lambda x: 'N' if 'N' in x else 'S')
    df['T/F'] = df['type'].apply(lambda x: 'T' if 'T' in x else 'F')
    df['J/P'] = df['type'].apply(lambda x: 'J' if 'J' in x else 'P')

    """Data cleaning"""

    #check the missing value
    df.isnull().sum()

    # Remove URLs and "|||" from the 'posts' column
    df['posts'] = df['posts'].apply(lambda x: re.sub(r'http\S+|www.\S+|\|\|\|', '', x))

    # remove punctuation
    import string
    string.punctuation
    def puntucation_free(text):
        output=''.join([i for i in text if i not in string.punctuation])
        return output
    df['posts']=df['posts'].apply(lambda x:puntucation_free(x))

    #Lower Text
    df['posts']=df['posts'].apply(lambda x:x.lower())

    # Tokenization
    from nltk.tokenize import word_tokenize
    df['posts']=df['posts'].apply(lambda x:word_tokenize(x))

    #remove stopwords
    import nltk
    nltk.download('stopwords')
    stopwords=nltk.corpus.stopwords.words('english')
    def remove_stopwords(text):
        output=[i for i in text if i not in stopwords]
        return output
    df['posts']=df['posts'].apply(lambda x:remove_stopwords(x))

    # stemming
    from nltk.stem.porter import PorterStemmer
    porter_stemmer=PorterStemmer()
    def stemming(text):
        output=[porter_stemmer.stem(i) for i in text]
        return output
    df['posts']=df['posts'].apply(lambda x:stemming(x))

    df['posts'] = df['posts'].str.join(' ')

    df.head(3)

    """Label encoding(I:1,E:0)(N:0,S:1)(F:0,T:1)(J:0,P:1)"""

    # le = LabelEncoder()
    # df['I/E'] = le.fit_transform(df['I/E'])
    # df['N/S'] = le.fit_transform(df['N/S'])
    # df['T/F'] = le.fit_transform(df['T/F'])
    # df['J/P'] = le.fit_transform(df['J/P'])

    # df.head()

    # """Full 16 classes classifier"""

    # preprocessing of the type column
    df['labels'] = pd.Categorical(df['type']).codes.astype(float) # .astype(float)
    # df['labels'] = df['labels'].apply(lambda x: np.eye(16, dtype=float)[x].tolist())


    print(df)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['posts'], df['labels'], test_size=0.2, stratify=df['labels'], random_state=42
    )

    # Tokenize the texts
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(texts):
        texts_list = texts.tolist()
        return tokenizer(texts_list, truncation=True, padding=True, max_length=1000)

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)

    train_dataset_full = MBTIDataset(train_encodings, train_labels)
    val_dataset_full = MBTIDataset(val_encodings, val_labels)

    # """Subclassifier I/E"""

    # # Train-validation split
    # train_texts, val_texts, train_labels, val_labels = train_test_split(
    #     df['posts'], df['I/E'], test_size=0.2, stratify=df['I/E'], random_state=42
    # )

    # # Tokenize the texts
    # tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # tokenizer.pad_token = tokenizer.eos_token

    # def tokenize_function(texts):
    #     texts_list = texts.tolist()
    #     return tokenizer(texts_list, truncation=True, padding=True, max_length=100)

    # train_encodings = tokenize_function(train_texts)
    # val_encodings = tokenize_function(val_texts)



    # train_dataset_ie = MBTIDataset(train_encodings, train_labels)
    # val_dataset_ie = MBTIDataset(val_encodings, val_labels)

    # """Subclassifier N/S"""

    # # Train-validation split
    # train_texts, val_texts, train_labels, val_labels = train_test_split(
    #     df['posts'], df['N/S'], test_size=0.2, stratify=df['N/S'], random_state=42
    # )
    # # Tokenize the texts
    # tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # tokenizer.pad_token = tokenizer.eos_token
    # train_encodings = tokenize_function(train_texts)
    # val_encodings = tokenize_function(val_texts)
    # # Convert to PyTorch datasets
    # train_dataset_ns = MBTIDataset(train_encodings, train_labels)
    # val_dataset_ns = MBTIDataset(val_encodings, val_labels)

    # """Subclassifier T/F"""

    # Train-validation split
    # train_texts, val_texts, train_labels, val_labels = train_test_split(
    #     df['posts'], df['T/F'], test_size=0.2, stratify=df['T/F'], random_state=42
    # )
    # # Tokenize the texts
    # tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # tokenizer.pad_token = tokenizer.eos_token
    # train_encodings = tokenize_function(train_texts)
    # val_encodings = tokenize_function(val_texts)

    
    # # Convert to PyTorch datasets
    # train_dataset_tf = MBTIDataset(train_encodings, train_labels)
    # val_dataset_tf = MBTIDataset(val_encodings, val_labels)

    # """Subclassifier J/P"""

    # # Train-validation split
    # train_texts, val_texts, train_labels, val_labels = train_test_split(
    #     df['posts'], df['J/P'], test_size=0.2, stratify=df['J/P'], random_state=42
    # )
    # # Tokenize the texts
    # tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # tokenizer.pad_token = tokenizer.eos_token
    # train_encodings = tokenize_function(train_texts)
    # val_encodings = tokenize_function(val_texts)
    # # Convert to PyTorch datasets
    # train_dataset_pj = MBTIDataset(train_encodings, train_labels)
    # val_dataset_pj = MBTIDataset(val_encodings, val_labels)

    # return train_dataset_full, train_dataset_ie, train_dataset_ns, train_dataset_tf, train_dataset_pj, val_dataset_ie, val_dataset_ns, val_dataset_tf, val_dataset_pj, val_dataset_full
    
    
    return train_dataset_full, val_dataset_full