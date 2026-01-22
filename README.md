# ml-ds
hi!, just uploading my jupyter files which contain the training and analysis of datasets that were done for AI/ML.
​
AI/ML Coursework – End‑to‑End ML Pipelines (Tabular, Image, Audio, Text)
Four Jupyter notebooks implementing full ML pipelines in Python, comparing classical models vs deep learning, with explainability and solid evaluation.
​

Project Structure
bash
├── Task1_Tabular.ipynb  # Tabular classification (XGBoost, TabNet, SHAP)
├── Task2_Image.ipynb    # Image classification (CNN vs classical, Grad‑CAM)
├── Task3_Sound.ipynb    # Audio classification (MFCC + CNN on mel-spectrograms)
└── Task4_Text.ipynb     # Fake vs real news (TF‑IDF vs DistilBERT + LIME/LDA)
Setup
bash
# Create environment (optional)
conda create -n aiml_coursework python=3.10
conda activate aiml_coursework

# Core dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tqdm

# Task‑specific
pip install xgboost shap tabnet-pytorch
pip install torch torchvision torchaudio
pip install librosa
pip install transformers lime gensim
Run any notebook:

bash
jupyter notebook Task1_Tabular.ipynb
Task 1 – Tabular Classification (Student/Structured Data)
​
Compares baseline models (Logistic Regression, Random Forest, XGBoost) vs a TabNet deep model on a structured dataset.

Includes preprocessing, feature scaling, class imbalance handling, full metrics, and SHAP for feature importance.

Result: XGBoost/TabNet achieve the highest F1 and ROC‑AUC, with SHAP highlighting a small set of key features driving predictions.

Task 2 – Image Classification (Vision CNN)
​
Image pipeline with classical baselines (e.g. k‑NN/SVM on features) vs a CNN in PyTorch.

Includes augmentation, train/val/test splits, confusion matrices, and Grad‑CAM style visualizations for interpretability.

Result: CNN clearly outperforms classical baselines; explainability shows the network focusing on salient object regions.

Task 3 – UrbanSound8K Audio Classification
​
Two‑track approach:

Classical ML (SVM, Random Forest, XGBoost) on MFCC, chroma, spectral contrast features.

CNN on mel‑spectrograms using PyTorch.

Includes SHAP on audio features and K‑Means clustering to explore unsupervised structure.

Result: Best classical model (SVM) reaches strong accuracy/F1; CNN on spectrograms performs best overall, with SHAP showing class‑specific frequency bands.

Task 4 – Fake vs Real News Text Classification
​
Classical text models: TF‑IDF + Logistic Regression, Naive Bayes, SVM.

Deep NLP: Fine‑tuned DistilBERT (HuggingFace Transformers) with a PyTorch training loop and scheduler.

Explainability via LIME for word‑level importance and LDA topic modeling for global themes.

Result: SVM already achieves high F1; DistilBERT reaches near‑perfect F1 on this dataset, with LIME showing fake news as more emotional/sensational and real news as more neutral.

Reproducibility and Design
All notebooks:

Set explicit random seeds for NumPy, torch, and sklearn.

Default to CPU but automatically use CUDA if available.

Use consistent plotting style and detailed metrics (accuracy, precision, recall, F1, ROC‑AUC, confusion matrices).

Intended as portfolio‑ready coursework for AI/ML, covering tabular, image, audio, and text modalities end‑to‑end.
