# TODO: Import necessary libraries for transformers and BERT model architecture
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# TODO: Define a function to load the pre-trained BERT model with a classification head
def create_model():
    # TODO: Load the pre-trained BERT-small model with weights
    model_name = "prajjwal1/bert-tiny"
    # TODO: Freeze some layers if needed to prevent overfitting
    # TODO: Add a Dense layer on top of BERT’s pooled output for classification (binary classification)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # TODO: Compile the model with binary cross-entropy loss and a suitable optimizer
    # TODO: Return the model
    return model,model_name

# TODO: Define a function to load a pre-trained BERT model and adapt it for sentiment analysis
def load_model():
    # TODO: Load the model from a specified file path or directory (model_path)
    model_name = './saved_model'
    # TODO: Ensure the loaded model matches the structure defined in create_model()
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # TODO: Return the loaded model
    return model,model_name
    pass



