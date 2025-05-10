# Keyphrase Generation and Stance Detection with BART

This folder (`Bartbase_Stance`) contains code for generating keyphrases from textual statements and detecting the stance (FAVOR, AGAINST, NONE) between the generated keyphrases and the original statements using a fine-tuned BERTweet model. The keyphrase generation is performed using the BART model (`facebook/bart-base`), and stance detection leverages a pre-trained or fine-tuned BERTweet model.

This guide explains how to set up the environment, execute the code, and understand the workflow. The repository also includes similar experiments with other models (e.g., T5, KeyBART) in their respective folders.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Repository Structure](#repository-structure)
4. [Setup Instructions](#setup-instructions)
5. [Data Preparation](#data-preparation)
6. [Executing the Keyphrase Generation Code](#executing-the-keyphrase-generation-code)
7. [Executing the Stance Detection Code](#executing-the-stance-detection-code)
8. [Output Files](#output-files)
9. [Troubleshooting](#troubleshooting)
10. [Additional Notes](#additional-notes)

## Overview
The workflow consists of two main tasks:
1. **Keyphrase Generation**: Uses the BART model to generate keyphrases from input statements (e.g., social media posts). The generated keyphrases are evaluated using metrics like F1, ROUGE, METEOR, BERTScore, YiSi, and MoverScore.
2. **Stance Detection**: Uses a fine-tuned BERTweet model to classify the stance (FAVOR, AGAINST, NONE) between the generated keyphrases and the original statements. Metrics such as precision and F1 score are computed.

The code supports processing multiple input CSV files, generating predictions, and saving results (predictions and metrics) to files.

## Prerequisites
- **Hardware**: A machine with a GPU (recommended for faster processing) and CUDA support if using GPU.
- **Operating System**: Windows, Linux, or macOS.
- **Python Version**: Python 3.8 or higher.
- **Dependencies**: Install the required Python packages listed in the [Setup Instructions](#setup-instructions).
- **Input Data**: CSV files containing the input statements and ground truth data (described in [Data Preparation](#data-preparation)).


- The `Bartbase_Stance` folder contains the code specific to experiments with the BART model.
-Similarly there are folders for t5 and kaybart ,However T5 explicitely dont have a seperate folder its in the main folder.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name/Bartbase_Stance

   pandas
## Install Dependencies:

numpy
torch
transformers
datasets
scikit-learn
rouge-score
nltk
bert-score

command : 
pip install -r requirements.txt   


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

Run the following Python commands to download required NLTK resources:

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

##Prepare the Fine-Tuned BERTweet Model:

The stance detection script uses a fine-tuned BERTweet model located at Stance\\bertweet_stance_finetuned .
Replace this path with the actual path to your fine-tuned BERTweet model, or use the pre-trained vinai/bertweet-base model by uncommenting the relevant lines in notebook.
To use your own fine-tuned model, place it in the models/bertweet_stance_finetuned folder and update the path in notebook.


##Data Preparation
The input data for both scripts should be in CSV format with the following columns:

For Keyphrase Generation (keyphrase_generation.py):
post: The input statement (e.g., social media post).
new_topic: The ground truth keyphrase for evaluation.

For Stance Detection (stance_detection.py):
post: The original statement.
predictions: The generated keyphrase (output from keyphrase_generation.py).
GT Stance: The ground truth stance label (FAVOR, AGAINST, NONE).
You can use the provided example datasets (tse_explicit.csv, tse_implicit.csv, vast_filtered_im.csv) or create your own. Place the CSV files in the data folder or update the file paths in the scripts.



##Executing the Keyphrase Generation Code
The keyphrase_generation.py script generates keyphrases from input statements using the BART model and evaluates them against ground truth keyphrases.

Update File Paths:
Open keyphrase_generation.py and update the test_files list with the paths to your input CSV files:
test_files = [
    "path/to/tse_explicit.csv",
    "path/to/tse_implicit.csv",
    "path/to/vast_filtered_im.csv"
]
Ensure the CSV files exist at the specified paths.

##What Happens:
The script loads the BART model (facebook/bart-base) and tokenizer.
For each input CSV file:
It processes the post column to generate keyphrases.
It computes evaluation metrics (F1, ROUGE-1, ROUGE-L, METEOR, BERTScore, YiSi, MoverScore) by comparing generated keyphrases to the new_topic column.
It saves the predictions to a CSV file (e.g., test_predictions_tse_explicit.csv) and metrics to a JSON file (e.g., test_metrics_tse_explicit.json).
#Generated files:
test_predictions_tse_explicit.csv: Contains predictions (generated keyphrases) and ground_truth (reference keyphrases).
test_metrics_tse_explicit.json: Contains evaluation metrics.
## Executing the Stance Detection Code
The stance_detection.py script predicts the stance (FAVOR, AGAINST, NONE) between the original statements and the generated keyphrases using a fine-tuned BERTweet model.

Update File Paths:
Open stance_detection.py and update the csv_files list with the paths to the prediction CSV files generated by keyphrase_generation.py:
 csv_files = [
    'test_predictions_tse_explicit.csv',
    'test_predictions_tse_implicit.csv',
    'test_predictions_vast_filtered_im.csv'
]
Update the BERTweet model path if necessary:
tokenizer = AutoTokenizer.from_pretrained("path/to/bertweet_stance_finetuned")
model = AutoModelForSequenceClassification.from_pretrained("path/to/bertweet_stance_finetuned")
##What Happens : 
What Happens:
The script loads the fine-tuned BERTweet model and tokenizer.
For each input CSV file:
It combines the post and predictions columns to predict the stance using BERTweet.
It adds a new column (predicted_stance_BERTTWEET) to the CSV file with the predicted stance.
It computes classification metrics (precision, F1 score, classification report) by comparing predicted stances to the GT Stance column.
It updates the input CSV file with the new predictions.
Finally, it computes and displays combined metrics across all files.
#Output Files
Keyphrase Generation:
test_predictions_<dataset>.csv: Contains generated keyphrases and ground truth keyphrases.
test_metrics_<dataset>.json: Contains evaluation metrics for each dataset.
Stance Detection:
Updated test_predictions_<dataset>.csv: Includes the predicted_stance_BERTTWEET column.
Console output with per-file and combined classification metrics.
