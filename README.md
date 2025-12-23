# SmartHealth Premium Predictor

A machine learning project that predicts health insurance premiums using demographic and medical data. The project uses age-based segmentation and advanced regression models to achieve high prediction accuracy and can be deployed for real-time premium estimation in applications.

---

## 🚀 Features
- End-to-end machine learning pipeline covering data ingestion, preprocessing, feature engineering, model training, evaluation, and deployment readiness. 
- Data preprocessing including handling missing values and outlier detection using IQR.  
- Feature engineering for both categorical and numerical variables to improve model performance.  
- Age-based segmentation:  
  - Young (<30)  
  - Rest (≥30)  
- Trained multiple regression models: Linear Regression, Ridge Regression, XGBoost, and XGBoost with hyperparameter tuning.  
- Achieved 98% accuracy (Ridge Regression) for Young group and 96.4% accuracy (XGBoost) for Rest group.  
- Models exported using Joblib for deployment in web or application environments.

---

## 🛠️ Tools & Technologies
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Joblib  

---

## 💻 Usage
1. Clone the repository:
```bash
git clone https://github.com/your-username/SmartHealth-Premium-Predictor.git
