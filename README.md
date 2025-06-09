# BERT Sentiment Analysis

This project demonstrates fine-tuning a pre-trained BERT model for binary sentiment analysis on the IMDb movie reviews dataset.

## Project Overview

The goal of this project is to classify movie reviews as either positive or negative using a fine-tuned BERT model. We leverage the `transformers` library from Hugging Face for easy access to pre-trained models and training utilities.

## Requirements

To run this project, you'll need the following libraries:

- `transformers`
- `datasets`
- `huggingface_hub`
- `fsspec`
- `torch`
- `pandas`
- `scikit-learn`

You can install these libraries using pip:

You might also need to install or upgrade `torchvision` depending on your environment and the CUDA version.

## Usage

The code is structured as a Jupyter notebook, but can be adapted for other environments. The main steps are:

1.  **Install Libraries:** Install the required Python libraries.
2.  **Load Dataset:** Load the IMDb dataset using the `datasets` library.
3.  **Tokenize Data:** Tokenize the movie reviews using the BERT tokenizer.
4.  **Load Pre-Trained BERT Model:** Load a pre-trained `bert-base-uncased` model for sequence classification.
5.  **Fine-Tune with Trainer:** Fine-tune the BERT model on the tokenized dataset using the `Trainer` API from `transformers`.
6.  **Evaluate the Model:** Evaluate the performance of the fine-tuned model using accuracy.
7.  **Save & Use the Fine-Tuned Model:** Save the fine-tuned model and tokenizer, and demonstrate how to load and use them for inference with the `pipeline` function.

## Code Structure

The code is presented sequentially in a Jupyter Notebook format (as seen in the provided context). Each step is clearly marked with a markdown header.

-   **Step 1: Install Libraries:** Includes the pip installation commands.
-   **Step 2: Load Dataset:** Loads the "imdb" dataset. Includes troubleshooting steps for potential `ValueError` when loading the dataset.
-   **Step 3: Tokenize the Data:** Defines a tokenization function and applies it to the dataset.
-   **Step 4: Load Pre-Trained BERT Model:** Loads the `BertForSequenceClassification` model.
-   **Step 5: Fine-Tune with Trainer:** Sets up `TrainingArguments` and the `Trainer` to fine-tune the model. Includes troubleshooting steps for potential issues during training.
-   **Step 6: Evaluate the Model:** Calculates and prints the accuracy of the model on the test set.
-   **Step 7: Save & Use the Fine-Tuned Model:** Saves the model and tokenizer and shows how to use a `pipeline` for inference.

## Troubleshooting

The provided code includes troubleshooting steps for common issues like `ValueError` during dataset loading. These steps involve checking internet connectivity, clearing the Hugging Face datasets cache, and upgrading relevant libraries.

## How to Run

1.  Open the provided code in a Jupyter notebook or an another environment like Jupyter Notebook.
2.  Run the code cells sequentially.
3.  Pay attention to the troubleshooting sections if you encounter errors.

## Results

After training and evaluation, the code will output the accuracy of the fine-tuned model on a subset of the test set. The final step demonstrates using the saved model to classify the sentiment of a new text input.

## Contributing

Feel free to contribute to this project by suggesting improvements, adding more features, or fixing bugs.

## License

MIT License Copyright (c) 2025 Dhanasekaran
