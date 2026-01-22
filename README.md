##### hi!, just uploading my jupyter files which contain the training and analysis of datasets that were done for AI/ML.

#### Four Jupyter notebooks in Python covering tabular, image, audio and text ML, with classical models and deep learning.

#### Project Structure
├── Task1_Tabular.ipynb  Tabular classification (XGBoost, TabNet, SHAP)

├── Task2_Image.ipynb    Image classification (CNN vs classical, Grad‑CAM)

├── Task3_Sound.ipynb    Audio classification (MFCC + CNN on mel-spectrograms)

└── Task4_Text.ipynb     Fake vs real news (TF‑IDF vs DistilBERT + LIME/LDA)




#### Task Summaries

Tabular: Classical baselines vs XGBoost/TabNet, with SHAP feature importance; boosted/deep models give best F1/ROC‑AUC.

Image: Classical feature‑based models vs CNN; CNN clearly wins, Grad‑CAM used for visual explanations.

Sound: SVM/RF/XGBoost on MFCC‑style features vs CNN on mel‑spectrograms; CNN achieves highest F1 on UrbanSound8K.

Text: TF‑IDF + classical models vs DistilBERT; SVM strong, DistilBERT reaches near‑perfect F1, with LIME and LDA for insight.
