# TODO: Import libraries and load functions from model.py and utils.py
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from model import create_model  # Import the function to initialize the model
from utils import load_data, preprocess_text, get_tokenizer,TqdmProgressBarCallback  # Import utility functions
import evaluate
import wandb


# wandb.init(project="sentiment_analysis")

# TODO: Call load_data() from utils.py to get preprocessed and tokenized data
train_dataset,test_dataset = load_data('./data/IMDB Dataset.csv')
# TODO: Initialize the model by calling create_model() from model.py
model, model_name= create_model()

tokenizer = get_tokenizer(model_name)

tokenized_train_dataset = train_dataset.map(lambda examples: preprocess_text(examples, tokenizer), batched=True)
tokenized_test_dataset = test_dataset.map(lambda examples: preprocess_text(examples, tokenizer), batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


# TODO: Train the model with fit, specifying the batch size, epochs, and validation data
# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    report_to= None,   # For Weights & Biases tracking
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
    # TODO: Set up callbacks if needed (e.g., early stopping)
# TODO: Save the trained BERT-based model

trainer.train()

trainer.save_model("./saved_model")
