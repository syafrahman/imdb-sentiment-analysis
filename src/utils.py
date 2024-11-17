# TODO: Import necessary libraries for data handling, BERT tokenization, and preprocessing
from datasets import load_dataset
# TODO: Import AutoTokenizer from transformers if needed for the tokenizer
from transformers import AutoTokenizer
from tqdm import tqdm

# TODO: Define a function to load and preprocess the IMDB dataset
def load_data(files_path):
    # TODO: Load the IMDB dataset from Hugging Face's datasets library
    dataset = load_dataset('csv',data_files=files_path)
    # TODO: Optionally, split the dataset into training and testing sets if not already split
    split_data = dataset['train'].train_test_split(test_size=0.2)
    train_dataset = split_data['train']
    test_dataset = split_data['test']

    # TODO: Return the dataset or split data as needed
    return train_dataset, test_dataset

def get_tokenizer(model_name):
    # TODO: Initialize the BERT tokenizer using the specified model name
    # (e.g., 'bert-small-uncased' or another appropriate model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def preprocess_text(examples, tokenizer):
    # TODO: Use the tokenizer to tokenize the input text
    # TODO: Pad or truncate sequences to the same length (e.g., 128 tokens) as required by the model
    # TODO: Convert tokenized text to tensors or the format required for model input
    tokenized_text = tokenizer(examples['review'], truncation=True, padding= False, max_length=128)
    tokenized_text["label"] = [1 if sentiment == "positive" else 0 for sentiment in examples["sentiment"]]
    # TODO: Return processed data suitable for input into the BERT model
    return tokenized_text

