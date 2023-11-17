## Classification with scikit-learn

### Dataset and Objective
The [project](https://github.com/JasonDahl/mlzoomcamp-homework/blob/main/homework-3/03-ml-classification-homework.ipynb "View project notebook") aims to explore a car price dataset, transforming the task into a classification problem by focusing on the 'MSRP' variable. The primary goal is to predict whether a car's price is above its mean value, creating a binary variable ('above_average').

### Libraries Utilized:
- **scikit-learn** for classification modeling and hyperparameter tuning.
- **NumPy** for numerical operations and data handling.
- **Pandas** for data manipulation and preprocessing.
- **Matplotlib** and **Seaborn** for data visualization.

### Data Preparation
- The dataset is downloaded and prepared, emphasizing specific columns such as 'Make,' 'Model,' 'Year,' 'Engine HP,' 'Engine Cylinders,' 'Transmission Type,' 'Vehicle Style,' 'highway MPG,' 'city MPG,' and 'MSRP' (transformed to 'price').
- Missing values in the selected features are handled by filling them with zeros.
- The data structure is examined to ensure proper formatting and completeness.

### Exploratory Analysis and Feature Correlation
- Identified the most frequent observation for the 'transmission_type' column ('AUTOMATIC' being the most common).
- Generated a correlation matrix for numerical features, revealing relationships among variables like 'engine_hp,' 'engine_cylinders,' 'highway_mpg,' and 'city_mpg.'
  ![Heatmap of Feature Correlation Matrix](https://github.com/JasonDahl/mlzoomcamp-homework/blob/main/homework-3/HW3_covar.png)

### Binary Classification Setup
- Transformed the 'price' variable into a binary format ('above_average') by categorizing prices above the mean as '1' and others as '0.'

### Data Splitting and Model Training
- Split the data into training, validation, and test sets following a 60%/20%/20% distribution, excluding the target ('above_average').
- Employed logistic regression, including categorical variables via one-hot encoding, trained on the training dataset.

### Model Evaluation and Feature Importance
- Calculated mutual information scores between 'above_average' and categorical variables ('make,' 'model,' 'transmission_type,' 'vehicle_style'), with 'transmission_type' having the lowest mutual information score.
- Conducted feature elimination to determine the least impactful feature, aiding in understanding features contributing less to the model's predictive power.

### Regression Analysis with Hyperparameter Tuning
- Applied a Ridge regression model to the 'price' column after a logarithmic transformation.
- Performed hyperparameter tuning for optimal model performance by testing different alpha values (regularization strength) to find the best RMSE score on the validation set.

### Conclusion
This project delves into comprehensive data analysis techniques, combining classification and regression methodologies to understand and predict car prices, leveraging libraries like scikit-learn, NumPy, Pandas, Matplotlib, and Seaborn. The findings showcase techniques applicable across diverse domains and industries.

### Potential Applications
The project's techniques hold relevance in various industries, including automotive retail, predictive modeling, pricing strategies, and feature importance understanding for classification tasks.
