# Model Evaluation

### Dataset and Objective:
This project focuses on model evaluation techniques - ROC AUC, precision, recall, and F1 score.  It starts with a car price dataset, transforming the task into a binary classification problem. The objective is to predict whether a car's price is above its mean value, creating a binary variable ('above_average'). Major techniques involve data preparation, exploratory analysis, model training using logistic regression, cross-validation, and hyperparameter tuning.

### Libraries Utilized:
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Data Preparation:
The notebook preprocesses data, selecting specific columns like 'Make,' 'Model,' 'Year,' 'Engine HP,' 'Engine Cylinders,' 'Transmission Type,' 'Vehicle Style,' 'highway MPG,' 'city MPG,' and 'MSRP' (transformed to 'price'). Missing values are handled by filling them with zeros, and the dataset is split into training, validation, and test sets.

### Exploratory Data Analysis:
The project explores feature importance by calculating the ROC AUC for each numerical variable concerning the 'above_average' target variable. ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) measures the model's ability to distinguish between classes, reflecting the trade-off between sensitivity and specificity. This analysis reveals influential features for predicting car prices.  In this case, 'Engine HP" is found to have the greatest importance for predicting MSRP.

### Model Training and Evaluation:
Logistic regression models are trained using one-hot encoding and evaluated using AUC on the validation dataset. Precision, recall, and F1 scores are computed to assess model performance.  Precision indicates the accuracy of positive predictions by measuring the proportion of correctly predicted positive cases among all predicted positives, while recall measures the completeness of positive predictions by calculating the proportion of correctly predicted positive cases among all actual positives.

$$Precision = \frac{True Positives}{True Positives + False Positives}$$

$$Recall = \frac{True Positives}{True Positives + False Negatives}$$

F1 Score represents the harmonic mean of precision and recall, providing a balanced measure that considers both false positives and false negatives in binary classification.

$$F1 Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

In this case, the precision-recall curves and the F1 score curve indicat an optimal decision probability threshold near 0.5:

![Precision, recall, and F1 curves](https://github.com/JasonDahl/mlzoomcamp-homework/blob/main/homework-4/HW4_F1.png)

### Model Cross-Validation and Hyperparameter Tuning:
Cross-validation techniques are employed for model optimization, assessing model performance across different folds of the dataset. Hyperparameter tuning, specifically for the regularization strength (C parameter), enhances the model's predictive capabilities.  Optimal C here is 10 - the highest value explored.  Exploring even larger values could lead to increased performance.

### Conclusion:
The project showcases robust data preprocessing, feature importance exploration for classification tasks, model training, and evaluation techniques using scikit-learn. Its applications include predictive modeling in various industries.
