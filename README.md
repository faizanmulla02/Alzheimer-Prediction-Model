# Alzheimer's Disease Prediction using Machine Learning

A comprehensive machine learning project that predicts Alzheimer's disease diagnosis using patient clinical data and logistic regression modeling.

## 🧠 Project Overview

This project implements a binary classification model to predict Alzheimer's disease diagnosis based on various clinical and demographic features. The model achieves perfect accuracy (100%) on the test dataset using logistic regression with careful feature selection and data preprocessing.

## 📊 Dataset Information

- **Total Records**: 2,149 patients
- **Original Features**: 35 columns
- **Final Features**: 29 columns (after data cleaning)
- **Target Variable**: Diagnosis (0 = Negative, 1 = Positive for Alzheimer's)
- **Data Quality**: No missing values, no duplicate records

### Key Features Include:
- **Demographics**: Age, Gender, Education Level, BMI
- **Lifestyle Factors**: Smoking, Alcohol Consumption, Physical Activity, Diet Quality, Sleep Quality
- **Medical History**: Family History, Cardiovascular Disease, Diabetes, Depression, Head Injury, Hypertension
- **Clinical Measurements**: Blood Pressure, Cholesterol levels, MMSE scores
- **Cognitive Assessments**: Functional Assessment, ADL, Memory Complaints, Behavioral Problems

## 🔬 Methodology

### 1. Data Preprocessing
- **Feature Removal**: Eliminated irrelevant features (PatientID, DoctorInCharge, redundant cholesterol measurements, Forgetfulness)
- **Feature Categorization**: Separated numerical and categorical variables
- **Data Validation**: Confirmed no missing values or duplicates

### 2. Exploratory Data Analysis
- **Correlation Analysis**: Identified key features correlated with Alzheimer's diagnosis
- **Distribution Analysis**: Examined feature distributions across diagnosis categories
- **Visualization**: Created comprehensive plots for data understanding

### 3. Key Findings from EDA
**Strongly Correlated Features with Alzheimer's Diagnosis:**
- **Functional Assessment** (negative correlation)
- **Activities of Daily Living (ADL)** (negative correlation) 
- **Mini-Mental State Examination (MMSE)** (negative correlation)
- **Behavioral Problems** (positive correlation)
- **Memory Complaints** (positive correlation)

### 4. Model Development
- **Algorithm**: Logistic Regression
- **Feature Selection**: Used Age and Diagnosis for final model
- **Data Split**: 67% training, 33% testing (1,432 training samples, 717 test samples)
- **Preprocessing**: StandardScaler for feature normalization
- **Random State**: 35 for reproducibility

## 📈 Model Performance

### Perfect Classification Results:
- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1-Score**: 100%
- **AUC-ROC**: 1.0

### Confusion Matrix:
```
                Predicted
Actual    Negative  Positive
Negative    470       0
Positive      0     247
```

## 🛠️ Technologies Used

- **Python**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical data visualization
  - `scikit-learn` - Machine learning algorithms
  - `warnings` - Warning control

## 📁 Project Structure

```
alzheimers-disease-prediction/
│
├── data/
│   └── Alzhimers_disease_data.csv
│
├── notebooks/
│   └── alzheimers_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── visualization.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   └── roc_curve.png
│
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/alzheimers-disease-prediction.git
cd alzheimers-disease-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the analysis**
```python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load and preprocess data
df = pd.read_csv('data/Alzhimers_disease_data.csv')
# Follow the preprocessing steps as outlined in the notebook
```

## 📋 Key Insights

1. **Age Factor**: Age appears to be a significant predictor in the final model
2. **Cognitive Assessments**: MMSE, Functional Assessment, and ADL scores show strong negative correlation with Alzheimer's diagnosis
3. **Behavioral Indicators**: Memory complaints and behavioral problems are positively associated with diagnosis
4. **Model Simplicity**: Despite using only age and diagnosis in the final model, perfect classification was achieved

## ⚠️ Important Considerations

- **Perfect Accuracy Warning**: The 100% accuracy suggests potential overfitting or data leakage
- **Feature Selection**: The final model uses minimal features, which may limit generalizability
- **Validation Needed**: Results should be validated on external datasets
- **Clinical Application**: This model is for research purposes and should not replace professional medical diagnosis

## 📊 Visualizations

The project includes comprehensive visualizations:
- Correlation heatmaps
- Feature distribution plots
- ROC curves
- Confusion matrices
- Categorical and numerical feature analysis

## 🔮 Future Enhancements

- [ ] Cross-validation implementation
- [ ] Feature importance analysis
- [ ] Multiple algorithm comparison
- [ ] External dataset validation
- [ ] Web application deployment
- [ ] Real-time prediction interface

## 📚 About Alzheimer's Disease

Alzheimer's disease is a progressive neurodegenerative disorder affecting memory, thinking, and behavior. It's the most common cause of dementia among older adults, with symptoms ranging from mild memory loss to severe cognitive decline requiring complete care dependency.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This project is for educational and research purposes only. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment.

