
# Drug Consumption Prediction Project

## 1. Introduction
This project focuses on analyzing and predicting drug consumption patterns based on psychological and demographic attributes using machine learning models.  
The dataset includes 18 types of drugs, where consumption frequency is encoded as ordinal categorical values (CL0–CL6). The goal is to predict the likelihood or class of consumption for each drug.

## 2. Dataset Overview
- **Dataset:** Drug Consumption (18 output features, 12 input features)
- **Input Features:** Age, Gender, Education, Country, Ethnicity, Nscore, Escore, Oscore, Ascore, Cscore, Impulsive, SS  
- **Output Features (Drugs):** Alcohol, Amphet, Amyl, Benzos, Caff, Cannabis, Choc, Coke, Crack, Ecstasy, Heroin, Ketamine, Legalh, LSD, Meth, Mushrooms, Nicotine, VSA  
- The original data labels (CL0–CL6) were converted to integer values (0–6) for modeling.

## 3. Methodology
All data processing and modeling were performed using **pandas** and **scikit-learn** libraries.

### a. Step 1 – Individual Drug Prediction (Multiclass Classification)
Two drugs — *Cannabis* and *Coke* — were selected as targets to train two different multiclass classification models:
1. Logistic Regression (for Cannabis and Coke)
2. Random Forest Classifier (for Cannabis and Coke)

#### Results:
| Model | Drug | Accuracy |
|--------|------|-----------|
| Logistic Regression | Cannabis | 0.416 |
| Logistic Regression | Coke | 0.568 |
| Random Forest | Cannabis | 0.414 |
| Random Forest | Coke | 0.581 |

**Interpretation:**  
Random Forest and Logistic Regression achieved moderate performance, with accuracies around 40–58%. Predicting drug consumption patterns from psychological features is challenging, suggesting non-linear and complex relationships among variables.

### b. Step 2 – Multi-Output Classification (16 Drugs)
The next step extended the analysis to all **16 remaining drugs** simultaneously using a **MultiOutputClassifier** with a **Random Forest** base model.

#### Results (Accuracy by Drug):
| Drug | Accuracy |
|------|-----------|
| Crack | 0.870 |
| Heroin | 0.854 |
| Ketamine | 0.804 |
| VSA | 0.772 |
| Meth | 0.756 |
| Amyl | 0.732 |
| Caff | 0.727 |
| Legalh | 0.647 |
| LSD | 0.634 |
| Ecstasy | 0.592 |
| Mushrooms | 0.536 |
| Benzos | 0.525 |
| Amphet | 0.504 |
| Choc | 0.424 |
| Alcohol | 0.385 |
| Nicotine | 0.361 |

**Interpretation:**  
The model performed best for drugs with highly imbalanced or distinct consumption patterns (e.g., Crack, Heroin). Substances with more common usage (e.g., Alcohol, Nicotine) were harder to predict, likely due to smaller variation in consumption classes.

## 4. Key Insights
- Multi-output Random Forest can handle multiple correlated targets efficiently.
- Psychological scores (Nscore, Escore, etc.) contribute significantly to predictive power.
- Drugs with clearer class boundaries show higher accuracy.
- Moderate overall accuracy indicates the complexity of behavioral data.

## 5. Future Improvements
- Apply hyperparameter tuning for better model optimization.
- Use feature selection or dimensionality reduction (PCA).
- Experiment with neural networks for non-linear relationships.
- Address class imbalance using oversampling or SMOTE.

## 6. Conclusion
This study successfully demonstrates multiclass and multi-output classification for predicting drug consumption based on psychological and demographic attributes. While some drugs were predicted with high accuracy, overall model performance highlights the complexity of behavioral data and the need for more advanced feature engineering.

---
*Prepared for: Machine Learning Coursework*  
*Environment: Microsoft Fabric (pandas, scikit-learn)*
