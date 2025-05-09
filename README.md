Automated Detection of Misinformation in News Media Using Advanced Natural Language Processing and Machine Learning Techniques
Author: Anirudh Singh
Registration Number: 22BCY10004
Course: GEN AI – NLP Assignment
Institution: VIT Bhopal
📌 Project Overview
This project presents a supervised machine learning solution for the classification of news articles as real or fake using Natural Language Processing (NLP) techniques. The model leverages the Fake and Real News Dataset from Kaggle and employs Logistic Regression for classification tasks. Emphasis has been placed on model interpretability, accessibility, and reproducibility, making it ideal for educational purposes. The model achieved a commendable accuracy of 92% and is supplemented with a Streamlit-based web interface for real-time prediction and visualization.
🧰 Technologies and Tools
- Programming Language: Python 3.8+
- Libraries Used:
  • pandas, numpy – Data preprocessing and manipulation
  • nltk – Text preprocessing (tokenization, stopword removal, lemmatization)
  • scikit-learn – Model training, evaluation, and TF-IDF vectorization
  • seaborn, matplotlib – Data visualization
  • Streamlit – Interactive user interface for live predictions
- Development Environment: Jupyter Notebook
📁 Repository Structure
- my_fake_news_detection.ipynb – Core implementation in Jupyter Notebook
- confusion_matrix.png – Visual representation of classification performance
- explanation.md – Phase 2 deliverable: Summary of the methodology
- report.tex – Phase 3 deliverable: Complete project report in LaTeX format
⚙️ Setup and Execution
1. Clone the Repository:
   git clone https://github.com/cryptoAnirudh/GenAi_Project_22BCY10004
   cd GenAi_Project_22BCY10004

2. Install Required Libraries:
   pip install pandas nltk scikit-learn numpy seaborn matplotlib

3. Download the Dataset:
   - Obtain the dataset from Kaggle:
     https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
   - Save the dataset files as follows:
     GenAi_Project_22BCY10004/data/Fake.csv
     GenAi_Project_22BCY10004/data/True.csv

4. Run the Notebook:
   - Launch Jupyter Notebook and open my_fake_news_detection.ipynb
   - Run all cells (Cell > Run All) to execute the full workflow
📊 Performance Metrics
- Overall Accuracy: 92%
- Classification Report:
  | Class | Precision | Recall | F1-Score |
  |-------|-----------|--------|----------|
  | Fake  | 0.91      | 0.93   | 0.92     |
  | Real  | 0.93      | 0.91   | 0.92     |

- Confusion Matrix: Available in confusion_matrix.png
- Example Prediction:
  Input: 'Government claims new policy boosts economy, lacks evidence.'
  Output: Fake
🔍 Key Features
- Implementation of a reliable and explainable Logistic Regression model
- Complete NLP pipeline: data cleaning, tokenization, lemmatization, and vectorization
- Real-time article classification via Streamlit web interface
- Lightweight design inspired by more complex BERT/LSTM architectures
- Developed for academic understanding and hands-on learning
🚀 Proposed Future Enhancements
- Incorporation of advanced NLP models (e.g., BERT, LSTM) for deeper semantic understanding
- Integration of live news feed APIs for real-time verification
- Expansion to include multilingual datasets for broader applicability
- Implementation of confidence scores and model interpretability tools (e.g., LIME, SHAP)
- Development of a mobile-friendly interface for wider accessibility
📚 Acknowledgments
- Dataset Source: Kaggle – Fake and Real News Dataset
- Conceptual inspiration drawn from advanced fake news detection workflows utilizing deep learning architectures, adapted here for simplicity and educational clarity.
