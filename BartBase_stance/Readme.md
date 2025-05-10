Keyphrase Generation and Stance Detection with BART
This folder (Bartbase_Stance) contains a Jupyter Notebook for generating keyphrases from textual statements and detecting the stance (FAVOR, AGAINST, NONE) between the generated keyphrases and the original statements using a fine-tuned BERTweet model. The keyphrase generation is performed using the BART model (facebook/bart-base), and stance detection leverages a pre-trained or fine-tuned BERTweet model. The code is organized in a single notebook (bart_stance_notebook.ipynb) with different cells for keyphrase generation and stance detection.
This guide explains how to set up the environment, execute the notebook, and understand the workflow. The repository also includes similar experiments with other models (e.g., T5, KeyBART) in their respective folders or the main directory, each with their own notebook and data.
Table of Contents

Overview
Prerequisites
Repository Structure
Setup Instructions
Data Preparation
Executing the Notebook
Output Files
Troubleshooting
Additional Notes

Overview
The workflow consists of two main tasks, implemented in separate cells within the same notebook:

Keyphrase Generation: Uses the BART model to generate keyphrases from input statements (e.g., social media posts). The generated keyphrases are evaluated using metrics like F1, ROUGE, METEOR, BERTScore, YiSi, and MoverScore.
Stance Detection: Uses a fine-tuned BERTweet model to classify the stance (FAVOR, AGAINST, NONE) between the generated keyphrases and the original statements. Metrics such as precision and F1 score are computed.

The notebook supports processing multiple input CSV files, generating predictions, and saving results (predictions and metrics) to files.
Prerequisites

Hardware: A machine with a GPU (recommended for faster processing) and CUDA support if using GPU.
Operating System: Windows, Linux, or macOS.
Python Version: Python 3.8 or higher.
Dependencies: Install the required Python packages listed in Setup Instructions.
Input Data: CSV files containing input statements and ground truth data, stored in the data subfolder of this folder (described in Data Preparation).
Jupyter Notebook: Required to run the notebook (bart_stance_notebook.ipynb).

Repository Structure
The repository is organized as follows:
├── Bartbase_Stance/
│   ├── bart_stance_notebook.ipynb       # Notebook for BART keyphrase generation and stance detection
│   ├── data/
│   │   ├── tse_explicit.csv            # Input dataset
│   │   ├── tse_implicit.csv            # Input dataset
│   │   ├── vast_filtered_im.csv        # Input dataset
│   ├── README.md                        # This guide
├── KeyBART_Stance/
│   ├── keybart_stance_notebook.ipynb    # Notebook for KeyBART experiments
│   ├── data/
│   │   ├── tse_explicit.csv            # Input dataset
│   │   ├── tse_implicit.csv            # Input dataset
│   │   ├── vast_filtered_im.csv        # Input dataset
│   ├── ...                              # Other filesягдти files
├── models/
│   ├── bertweet_stance_finetuned/       # Fine-tuned BERTweet model for stance detection
├── t5_stance_notebook.ipynb             # Notebook for T5 experiments (in main folder)
├── data/
│   ├── tse_explicit.csv                # Input dataset
│   ├── tse_implicit.csv                # Input dataset
│   ├── vast_filtered_im.csv            # Input dataset
└── README.md                            # Main repository README


The Bartbase_Stance folder contains the BART experiment notebook and its own data subfolder for input CSV files.
The KeyBART_Stance folder contains a similar notebook and data subfolder for KeyBART experiments.
T5-related experiments are in the main repository folder (not a separate subfolder) with their own notebook and data subfolder.
The models folder contains the fine-tuned BERTweet model for stance detection.

Setup Instructions

Clone the Repository:
git clone https://github.com/your-repo-name.git
cd your-repo-name/Bartbase_Stance


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Create a requirements.txt file with the following content:
pandas
numpy
torch
transformers
datasets
scikit-learn
rouge-score
nltk
bert-score
jupyter

Install the dependencies:
pip install -r requirements.txt


Download NLTK Data:Run the following Python commands to download required NLTK resources:
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


Prepare the Fine-Tuned BERTweet Model:

The stance detection cells use a fine-tuned BERTweet model located at ../models/bertweet_stance_finetuned (relative to the notebook).
Replace this path with the actual path to your fine-tuned BERTweet model, or use the pre-trained vinai/bertweet-base model by updating the relevant cell in the notebook:tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=3)


To use your own fine-tuned model, place it in the models/bertweet_stance_finetuned folder and ensure the path in the notebook is correct.


Verify GPU Availability (optional):If you have a GPU, ensure CUDA is installed and verify that PyTorch detects it:
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available



Data Preparation
The input data should be in CSV format and stored in the data subfolder of the Bartbase_Stance folder (Bartbase_Stance/data/). The CSV files must have the following columns:

For Keyphrase Generation (in the keyphrase generation cells):

post: The input statement (e.g., social media post).
new_topic: The ground truth keyphrase for evaluation.
Example:post,new_topic
"I love renewable energy!","renewable energy"
"Climate change is a hoax","climate change"




For Stance Detection (in the stance detection cells):

post: The original statement.
predictions: The generated keyphrase (output from the keyphrase generation cells).
GT Stance: The ground truth stance label (FAVOR, AGAINST, NONE).
Example:post,predictions,GT Stance
"I love renewable energy!","renewable energy",FAVOR
"Climate change is a hoax","climate change",AGAINST





You can use the provided example datasets (tse_explicit.csv, tse_implicit.csv, vast_filtered_im.csv) or create your own. Place the CSV files in Bartbase_Stance/data/ and update the file paths in the notebook cells if necessary.
Executing the Notebook
The bart_stance_notebook.ipynb notebook contains cells for both keyphrase generation and stance detection.

Start Jupyter Notebook:
jupyter notebook

Open bart_stance_notebook.ipynb in the browser.

Update File Paths:

In the keyphrase generation cells, update the test_files list with the paths to your input CSV files in the data subfolder:test_files = [
    "data/tse_explicit.csv",
    "data/tse_implicit.csv",
    "data/vast_filtered_im.csv"
]


In the stance detection cells, update the csv_files list with the paths to the prediction CSV files generated by the keyphrase generation cells:csv_files = [
    "data/test_predictions_tse_explicit.csv",
    "data/test_predictions_tse_implicit.csv",
    "data/test_predictions_vast_filtered_im.csv"
]


Update the BERTweet model path in the stance detection cells if necessary:tokenizer = AutoTokenizer.from_pretrained("../models/bertweet_stance_finetuned")
model = AutoModelForSequenceClassification.from_pretrained("../models/bertweet_stance_finetuned")




Run the Notebook:

Execute the cells in sequence (use Shift + Enter to run each cell).
Start with the cells for keyphrase generation, which will generate keyphrases and save predictions to CSV files in the data subfolder.
Then run the stance detection cells, which will predict stances and update the prediction CSV files.


What Happens:

Keyphrase Generation Cells:
Load the BART model (facebook/bart-base) and tokenizer.
For each input CSV file in test_files:
Process the post column to generate keyphrases.
Compute evaluation metrics (F1, ROUGE-1, ROUGE-L, METEOR, BERTScore, YiSi, MoverScore) by comparing generated keyphrases to the new_topic column.
Save predictions to a CSV file (e.g., data/test_predictions_tse_explicit.csv) and metrics to a JSON file (e.g., data/test_metrics_tse_explicit.json).




Stance Detection Cells:
Load the fine-tuned BERTweet model and tokenizer.
For each input CSV file in csv_files:
Combine the post and predictions columns to predict the stance.
Add a predicted_stance_BERTTWEET column to the CSV file.
Compute classification metrics (precision, F1 score, classification report) by comparing predicted stances to the GT Stance column.
Update the input CSV file with the new predictions.


Display combined metrics across all files.




Example Output:

Console/notebook output for keyphrase generation:Using device: cuda
Loading tokenizer and model from facebook/bart-base...
🔍 Testing on: data/tse_explicit.csv
✅ Saved: data/test_predictions_tse_explicit.csv, data/test_metrics_tse_explicit.json


Console/notebook output for stance detection:📂 Processing file: data/test_predictions_tse_explicit.csv
✅ Saved updated file with predictions to: data/test_predictions_tse_explicit.csv
📊 Metrics for: data/test_predictions_tse_explicit.csv
...
🧮 Overall Combined Results:
...


Generated files in data/:
test_predictions_<dataset>.csv: Contains generated keyphrases, ground truth keyphrases, and (after stance detection) predicted stances.
test_metrics_<dataset>.json: Contains evaluation metrics for keyphrase generation.





Output Files
All output files are saved in the data subfolder (Bartbase_Stance/data/):

Keyphrase Generation:
test_predictions_<dataset>.csv: Contains generated keyphrases and ground truth keyphrases.
test_metrics_<dataset>.json: Contains evaluation metrics for each dataset.


Stance Detection:
Updated test_predictions_<dataset>.csv: Includes the predicted_stance_BERTTWEET column.
Classification metrics displayed in the notebook output.



Troubleshooting

FileNotFoundError: Ensure the input CSV files exist in Bartbase_Stance/data/. Update the paths in the notebook cells if necessary.
ModuleNotFoundError: Verify that all dependencies are installed (pip install -r requirements.txt).
CUDA Out of Memory: Reduce the batch size or run on CPU by setting device = torch.device("cpu") in the relevant cells.
BERTweet Model Path Error: Ensure the fine-tuned BERTweet model path is correct or use the pre-trained vinai/bertweet-base model.
Encoding Issues: If CSV files fail to load, ensure they are encoded in ISO-8859-1 or update the encoding in pd.read_csv (e.g., encoding="ISO-8859-1").
Jupyter Notebook Not Opening: Ensure Jupyter is installed (pip install jupyter) and run jupyter notebook from the correct directory.

Additional Notes

T5 Experiments: The T5 notebook (t5_stance_notebook.ipynb) is in the main repository folder, with its own data subfolder. Follow similar setup and execution steps, updating model-specific paths and configurations.
KeyBART Experiments: The KeyBART_Stance folder contains a notebook (keybart_stance_notebook.ipynb) and data subfolder, following a similar structure to this folder.
Customizing Metrics: Modify the calculate_metrics function in the keyphrase generation cells to add or remove evaluation metrics.
Fine-Tuning BERTweet: To fine-tune the BERTweet model, refer to the Hugging Face Transformers documentation for sequence classification tasks.
Scalability: For large datasets, consider batching the input data or using a more powerful GPU.
Notebook Execution Order: Ensure you run the keyphrase generation cells before the stance detection cells, as the latter depend on the generated prediction CSV files.

For further assistance, open an issue in the repository or contact the repository maintainers.
