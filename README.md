# Abstract
Predicting the tags of competitive programming problems can guide learning and problem-solving, but current research focuses on Codeforces or other platforms rather than the USA Computing Olympiad (USACO), which has a unique problem writing style with high narrative fluff rather than explicit tasks or technical language. Previous research also has trained predominantly on text-only features, and has generally not gone beyond model training into Explainable AI (XAI). This study trained models to predict USACO tags, testing the effects of including explicit numeric constraint features and applying XAI to interpret predictions. We curated 441 tagged USACO problems from the USACO Guide repository, extracting numeric constraints from within problem statements. TF-IDF based linear models (LinearSVM, Logistic Regression) were compared with deep-learning models (SciBERT, CodeBERT, DeBERTa, SetFit), and compared with and without numeric constraint features. XAI with linear feature weights and SHAP analysis investigated genuine algorithmic insights and signs of overfitting within models. We found that Logistic Regression with numeric constraints performed best (Macro F1 = 0.3034, Exact Match Accuracy = 0.075). Explicitly incorporating constraint features was shown to consistently boost performance. SHAP analysis revealed deep-learning models may have been confused by narrative filler rather than isolating key terms like linear models. These findings suggest linear models can outperform deep learning on small datasets with distracting narrative content. This work contributes the largest known tagged USACO dataset and supports developing human-in-the-loop tagging tools for competitive programming and real-world programming tasks.

# Overview of Methodology
<p align="center">
<img width="46%"  alt="Data Sourcing and Tagging" src="https://github.com/user-attachments/assets/a1261ba5-8e21-4dbf-b89b-1683e43a1fad" /> <img width="44%"  alt="TF-IDF + Numeric Constraints Training (1)" src="https://github.com/user-attachments/assets/9d2cae4a-f89c-4d19-a2b7-a71c8ab76a6f" />
</p>

**Figure 1 | Data sourcing, filtering, and tagging workflow.**        
**Figure 2 | TF-IDF and Numeric Feature Extraction for linear model training workflow**


Full manuscript in preparation
