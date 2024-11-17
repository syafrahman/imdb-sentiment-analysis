# TODO: Import libraries for predictions and load functions
from transformers import AutoTokenizer
import torch
from model import load_model

# TODO: Load the saved BERT model
model, model_name = load_model()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# TODO: Define a function to preprocess new reviews for prediction
def preprocess_text(text):
    # TODO: Tokenize and pad the new review text to match the model’s input requirements
    tokenized_text = tokenizer(text, truncation=True, padding= False, max_length=128, return_tensors ='pt')
    return tokenized_text

# TODO: Define a function to make predictions on new data
def predict(text):
    # TODO: Use the model’s predict function on the preprocessed input and interpret results
    inputs=preprocess_text(text)

    with torch.no_grad():
        outputs=model(**inputs)
        logits = outputs.logits

        # Post-process to get the predicted label
        predicted_class = torch.argmax(logits, dim=-1).item()
        label_mapping = {0: "negative", 1: "positive"}  # Adjust based on your labels
        return label_mapping[predicted_class]

    # Example usage
if __name__ == "__main__":
    text = "This movie was great!"
    print(f"Prediction: {predict(text)}")
