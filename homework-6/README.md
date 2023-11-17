## Decision Trees and Ensemble Learning

### Dataset and Objective

The project centers on the California Housing Prices dataset sourced from Kaggle, with the primary aim of predicting 'median_house_value.' It employs ensemble learning techniques like Decision Trees, Random Forests, and XGBoost for regression analysis.

#### Utilizing Ensemble Models

Ensemble learning combines multiple models to enhance predictive performance. In this context, ensemble models, particularly Random Forests, amalgamate predictions from various decision trees to yield robust and accurate outcomes. Additionally, the integration of XGBoost, a gradient boosting algorithm, bolsters performance by iteratively refining models' shortcomings.

#### Focus on Model Tuning

Model tuning is a crucial aspect of this project. Tuning involves adjusting hyperparameters like the number of trees (n_estimators) and tree depth (max_depth) to optimize model performance. The process entails evaluating numerous parameter configurations to identify the most effective setup, emphasizing metrics like precision, recall, or, in this case, RMSE (Root Mean Square Error) for regression tasks.


### Libraries Utilized
The [Jupyter notebook](https://github.com/JasonDahl/mlzoomcamp-homework/blob/main/homework-6/06-trees-homework.ipynb "View notebook") employs pandas, numpy, matplotlib, seaborn, and scikit-learn for data processing, model building, and evaluation. Additionally, it incorporates XGBoost for advanced ensemble learning.

### Dataset Preparation

The "Dataset Preparation" section aims to ready the California Housing Prices dataset for effective model training and assessment. It encompasses several essential steps:

1. **Subset Selection:** Filtering the dataset to retain records with 'ocean_proximity' values '<1H OCEAN' or 'INLAND'. This reduction narrows the data focus for analysis.

2. **Handling Missing Values:** Managing missing data by filling NaN values in specific columns, particularly 'total_bedrooms,' with zeros. This strategy ensures continuity in the analysis by addressing null values.

3. **Target Variable Transformation:** Applying a logarithmic transformation (log1p) to the 'median_house_value' column. This transformation can normalize the distribution of this variable, often resulting in more stable model performance.

4. **Train-Validation-Test Split:** Segmenting the data into three subsets—training, validation, and testing—using 'train_test_split.' This allocation allows for model training, tuning, and final evaluation on distinct datasets, ensuring unbiased performance assessment.

5. **Data Encoding:** Employing DictVectorizer to convert the segmented datasets (training, validation, and test) into matrix format. This step is crucial for model compatibility and application of machine learning algorithms.

This section ensures that the dataset is appropriately structured and processed for subsequent model training, evaluation, and validation. It emphasizes data cleanliness, subset selection, target variable transformation, and data encoding to support the model-building process.

### Model Training

The section covers the training of two regression models: a Decision Tree Regressor and a Random Forest Regressor, both focused on predicting the 'median_house_value' variable.

#### Decision Tree Regressor
The Decision Tree Regressor was trained with a maximum depth of 1. It utilized the 'ocean_proximity' feature to split the dataset into two subsets: '<1H OCEAN <= 0.50' and '<1H OCEAN > 0.50,' predicting values of '11.61' and '12.30,' respectively. This straightforward split highlights the primary decision criterion for the model's performance.

#### Random Forest Regressor
Following that, a Random Forest Regressor was trained with 10 estimators and a random state of 1. Its validation RMSE stood at approximately 0.245, representing the average prediction error concerning the true values in the validation dataset.

Comparison of the models reveals a trade-off: the Decision Tree Regressor, simplistic with its 'ocean_proximity' based splitting, versus the Random Forest Regressor, which, despite its ensemble nature, exhibits higher predictive power albeit with increased complexity. The latter's RMSE of 0.245 on validation suggests stronger predictive ability compared to the simpler Decision Tree.

### Hyperparameter Tuning: n_estimators and max_depth

This section explores optimizing the Random Forest Regressor by tuning two crucial hyperparameters: `n_estimators` and `max_depth`.

#### Experimenting with n_estimators
The `n_estimators` parameter, determining the number of trees in the forest, was varied from 10 to 200 in increments of 10. The model's performance was evaluated on the validation dataset, measuring RMSE for each configuration.

The plot of RMSE against the number of estimators revealed diminishing improvements after reaching an `n_estimators` value of 160. At this point, the RMSE stabilized around 0.234, indicating no significant reduction beyond this threshold.

![RMSE for different values of 'n_estimators'](https://github.com/JasonDahl/mlzoomcamp-homework/blob/main/homework-6/HW6_n_estimators.png)

#### Selecting the best max_depth
Next, different values of `max_depth` (10, 15, 20, 25) were experimented with, in combination with varying `n_estimators` from 10 to 200. The mean RMSE was calculated for each `max_depth` and `n_estimators` pair.

Observations showcased that as `max_depth` increased, the mean RMSE decreased. The mean RMSE values were recorded as 0.2454 for `max_depth=10`, 0.2359 for `max_depth=15`, 0.2353 for `max_depth=20`, and 0.2349 for `max_depth=25`.

Among these options, `max_depth=25` demonstrated the lowest mean RMSE, indicating superior predictive performance. The heatmap visualization of RMSE values for different combinations reaffirmed that `max_depth=25` with varying `n_estimators` consistently led to lower RMSE scores, suggesting its suitability as the optimal `max_depth` choice for this model configuration.

![RMSE heatmap for different values of 'n_estimators' and 'max_depth](https://github.com/JasonDahl/mlzoomcamp-homework/blob/main/homework-6/HW6_RMSE.png)

### Feature Importance Exploration

Understanding which features have the most significant impact on a model's predictions is crucial. Tree-based models like Random Forest Regressors provide a valuable metric called "feature importance", which helps identify the most influential predictors.

#### Extracting Feature Importance Information
In tree-based models, such as Random Forest Regressors, the feature importance information is available in the `feature_importances_` attribute. For this analysis, a Random Forest Regressor model was trained with specific parameters: `n_estimators=10`, `max_depth=20`, `random_state=1`.

The feature importance scores were extracted from this model, revealing the following insights:

1. **Median Income** stood out as the most influential feature, with an importance score of approximately 0.336. This metric signifies that variations in median income significantly impact predictions.

2. **Proximity to Ocean (<1H OCEAN)** emerged as the second most important feature, demonstrating a considerable impact, with an importance score of approximately 0.292.

3. **Latitude and Longitude** also exhibited notable importance, although comparatively less influential than median income and ocean proximity. Latitude accounted for approximately 0.102 importance, while longitude contributed around 0.086 importance.

4. **Inland Ocean Proximity** was another relevant feature, though with less impact than the top three, registering an importance score of approximately 0.074.

Other features such as `housing_median_age`, `population`, `total_rooms`, `total_bedrooms`, and `households` displayed comparatively lower importance scores in this analysis.

Among the provided features, the analysis identified **Median Income** as the most crucial predictor, indicating that changes in median income strongly influence the model's predictions.

### XGBoost Model

XGBoost, a popular gradient boosting algorithm, provides various hyperparameters for fine-tuning models. Among these, the `eta` parameter (also known as learning rate) significantly influences the model's performance.

#### Eta Parameter Exploration
The XGBoost model was trained with differing values for the `eta` parameter to discern its impact on model performance.

##### Initial Model Training
The first model was trained with an `eta` of 0.3. After 100 boosting rounds, the model demonstrated a consistent improvement in performance:
- The RMSE on the training set started at approximately 0.4435 and gradually decreased to 0.1099.
- On the validation dataset, the RMSE also exhibited a notable decrease from around 0.4425 to 0.2286.

##### Further Eta Tuning
The `eta` value was adjusted to 0.1 for subsequent training. This adjustment led to a different model performance:
- The RMSE on the training set began at about 0.5245, dropping to approximately 0.1632 after 100 boosting rounds.
- Validation dataset RMSE similarly decreased from around 0.5205 to 0.2321.

#### Selecting the Best Eta
Comparing the two scenarios, the model trained with an `eta` of 0.3 achieved a lower RMSE of about 0.2286 on the validation dataset, whereas the model with an `eta` of 0.1 resulted in an RMSE of approximately 0.2321.

Therefore, based on the evaluation on the validation dataset, an `eta` value of 0.3 led to a better RMSE score, indicating superior performance for this specific dataset and modeling scenario.

### Conclusion
The project demonstrates the process of building regression models, focusing on Decision Trees, Random Forests, and XGBoost. It covers dataset preparation, model training, tuning, feature importance analysis, and XGBoost training, showcasing ensemble learning techniques' versatility for regression tasks.
