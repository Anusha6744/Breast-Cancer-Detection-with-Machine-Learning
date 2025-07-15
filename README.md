Breast Cancer Detection Using Machine Learning

This project predicts whether a breast tumor is **malignant** or **benign** using diagnostic data. It applies supervised machine learning algorithms such as **Logistic Regression**, **Random Forest**, **Decision Tree** and **Support Vector Machine (SVM)** with performance evaluation using metrics like accuracy, confusion matrix, and classification report.

---

##  Dataset

- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Records**: 569 patient samples
- **Features**: 30 numeric features (e.g., mean radius, texture, perimeter, concavity)
- **Target**:
  - `0` → Malignant (cancerous)
  - `1` → Benign (non-cancerous)

---

##  Objective

To build and evaluate machine learning models that accurately classify breast tumors based on clinical features as malignant or benign.

---

##  Technologies Used

- **Python**
- **Libraries**:
  - `pandas`, `numpy` for data handling  
  - `seaborn`, `matplotlib` for visualization  
  - `scikit-learn` for machine learning models and metrics  
- **Preprocessing**: `StandardScaler`
- **Model Validation**: Train-test split

---

##  Project Workflow

### 1. Data Exploration
- Displayed dataset head, info, and statistics
- Checked data types, class distribution, and missing values

### 2. Data Visualization
- Visualized target class distribution using `sns.countplot`
- Generated a correlation heatmap to study feature relationships

### 3. Preprocessing
- Scaled features using `StandardScaler` for better model performance
- Performed an 80-20 train-test split

### 4. Model Training
Trained and evaluated the following models:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**

**Evaluation Metrics**:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

### 5. Feature Importance
- Used Random Forest’s feature importance to identify key predictors:
  - **Worst perimeter**
  - **Mean concavity**
  - **Worst concave points**
  - **Mean area**

---

## 📈 Results

| Model               | Accuracy (Test Set) |
|---------------------|---------------------|
| Logistic Regression | 97%                 |
| Decision Tree       | 95%
| Random Forest       | 96%                 |
| SVM                 | 98%                 |

---

## 💡 Key Insights

- All models achieved high accuracy, with **SVM performing best** on the test set.
- **Feature scaling** using `StandardScaler` significantly improved performance.
- Certain features like **worst concave points** and **mean area** strongly influence tumor classification.
- Even simple models like Logistic Regression perform very well on this dataset.

---

##  Future Improvements

- Apply cross-validation to check model stability
- Tune model hyperparameters using `GridSearchCV`
- Save the best model using `joblib` or `pickle`
- Build a simple web app using Streamlit or Flask
- Add SHAP or LIME to interpret predictions

---

##  Author

**Your Name**  
📧 aajayan525@gmail.com 
🔗 [GitHub](https://github.com/Anusha6744)

---

## License

This project is for educational use. Dataset provided by `scikit-learn`.

---

## ⭐️ Show Support

If you liked this project, consider ⭐️ starring the repository or sharing it with others!
