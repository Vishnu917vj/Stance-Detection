🔑 Keyphrase Generation and Stance Detection
This project performs Keyphrase Generation using a BART-based sequence-to-sequence model and evaluates the generated keyphrases using various NLP metrics. It also includes Stance Detection between the generated keyphrases and the original posts using a fine-tuned BERTweet model.

📁 Project Structure
bash
Copy
Edit
project-root/
│
├── keyphrase_generation.py      # Code for generating keyphrases and computing metrics
├── stance_detection.py          # Code for predicting stance using BERTweet
├── test_predictions_*.csv       # Output prediction files
├── test_metrics_*.json          # Evaluation metrics
└── README.md                    # Project documentation
🧠 Features
1. Keyphrase Generation
Uses a pre-trained or fine-tuned facebook/bart-base model.

Input: Text posts (from CSV files).

Output: Generated keyphrases.

Evaluates using:

F1-score

ROUGE-1, ROUGE-L

METEOR

BERTScore

YiSi

MoverScore

2. Stance Detection
Uses a fine-tuned BERTweet model for stance classification (FAVOR, AGAINST, NONE).

Input: Combination of post and generated keyphrase.

Output: Stance prediction.

Evaluates using:

Classification Report

Precision (Macro)

F1 Score (Macro)

🔧 Installation
Requirements
Python 3.8+

transformers

datasets

torch

nltk

pandas

scikit-learn

rouge-score

bert-score

Install Dependencies
bash
Copy
Edit
pip install transformers datasets torch nltk pandas scikit-learn rouge-score bert-score
Download NLTK Data
python
Copy
Edit
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
🚀 Usage
1. Run Keyphrase Generation
Update the test file paths in keyphrase_generation.py:

python
Copy
Edit
test_files = [
    "path/to/tse_implicit.csv",
    "path/to/vast_filtered_im.csv",
    "path/to/tse_explicit.csv"
]
Then run:

bash
Copy
Edit
python keyphrase_generation.py
It will:

Generate keyphrases for each post.

Save predictions to test_predictions_<dataset>.csv.

Save evaluation metrics to test_metrics_<dataset>.json.

2. Run Stance Detection
Ensure your model is fine-tuned and available locally or from Hugging Face.

Update paths to your prediction CSVs and model directory in stance_detection.py.

Then run:

bash
Copy
Edit
python stance_detection.py
It will:

Predict stances for each post-keyphrase pair.

Save updated CSVs with a new column: predicted_stance_ BERTTWEET.

Print classification reports and macro precision/F1.

📊 Example Output
🔑 Keyphrase Generation Metrics
json
Copy
Edit
{
    "F1": 0.742,
    "ROUGE-1": 0.615,
    "ROUGE-L": 0.591,
    "METEOR": 0.433,
    "BERTScore": 0.812,
    "YiSi": 0.789,
    "MoverScore": 0.755
}
🧭 Stance Detection Metrics
text
Copy
Edit
              precision    recall  f1-score   support

       FAVOR       0.78      0.75      0.76       100
     AGAINST       0.70      0.72      0.71       90
         NONE       0.69      0.70      0.69       85

   macro avg       0.72      0.72      0.72       275
🧪 Model Details
Keyphrase Generation Model
facebook/bart-base

Fine-tuning recommended on your own dataset for better results.

Stance Detection Model
Fine-tuned vinai/bertweet-base on stance-labeled data (3 classes: FAVOR, AGAINST, NONE).

Input format: "post [SEP] keyphrase"

📝 License
This project is for educational and research purposes. Always cite relevant model sources (e.g., BART, BERTweet) when publishing results.

🙋‍♂️ Acknowledgements
Facebook BART

BERTweet by VinAI

Hugging Face Transformers

[ROUGE, METEOR, BERTScore, YiSi, MoverScore papers and tools]
