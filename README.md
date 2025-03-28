# House Price Prediction Using the Boston Housing Dataset

## Project Overview
This project aims to predict house prices in the Boston area using the **Boston Housing dataset**. By leveraging machine learning techniques, specifically a **RandomForestRegressor**, the model predicts the median house price based on various features such as crime rate, number of rooms, and proximity to employment centers. The project includes data exploration, model training, evaluation, and a feature for making predictions on new data.

---

## Dataset Information
The Boston Housing dataset is a classic dataset for regression tasks and contains information about housing in the Boston area. It includes the following:

- **Features (13 in total)**:
  - CRIM: Per capita crime rate by town
  - ZN: Proportion of residential land zoned for large lots
  - INDUS: Proportion of non-retail business acres per town
  - CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
  - NOX: Nitric oxides concentration (air pollution)
  - RM: Average number of rooms per dwelling
  - AGE: Proportion of owner-occupied units built before 1940
  - DIS: Weighted distances to five Boston employment centers
  - RAD: Index of accessibility to radial highways
  - TAX: Full-value property tax rate per $10,000
  - PTRATIO: Pupil-teacher ratio by town
  - B: Measure related to the proportion of Black residents
  - LSTAT: Percentage of lower-status population

- **Target Variable**:
  - PRICE: Median value of owner-occupied homes in $1000s

### Data Preprocessing
- **Missing Values**: The dataset has no missing values.
- **Feature Engineering**: A new feature, `RM_TAX_RATIO` (ratio of rooms to tax rate), was created to capture potential interactions between these variables.
- **Log Transformation**: A logarithmic version of the target (`LogPRICE`) was created but not used in the final model.

---

## Installation and Dependencies
To run this project, you need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`

You can install them using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## Usage Instructions
1. **Load the Dataset**: The dataset is loaded from the official source using `pd.read_csv`. The data is then processed into a DataFrame for easy manipulation.
2. **Explore the Data**: Visualizations such as histograms, scatter plots, and a correlation heatmap are generated to understand the relationships between features and the target variable.
3. **Prepare the Data**: The dataset is split into training (80%) and testing (20%) sets using `train_test_split`.
4. **Train the Model**: A RandomForestRegressor with 100 trees is trained on the training data.
5. **Evaluate the Model**: The model's performance is evaluated using Root Mean Squared Error (RMSE) and R² score on the test set.
6. **Make Predictions**: The model can predict house prices for new data, either from the test set or user-provided inputs.
7. **Save the Model**: The trained model is saved using `joblib` for future use.

To run the entire project, execute the Python script:
```bash
python house_price_prediction.py
```

---

## Model Details
- **Algorithm**: RandomForestRegressor
- **Hyperparameters**: 
  - `n_estimators=100`
  - `random_state=42`
- **Why RandomForest?**: Random forests handle non-linear relationships and interactions between features effectively, making them suitable for this regression task.

### Evaluation Metrics
- **Root Mean Squared Error (RMSE)**: $2,730 (indicating the average prediction error in house prices).
- **R² Score**: 0.90 (meaning the model explains 90% of the variance in house prices).

These metrics demonstrate that the model performs well in predicting house prices based on the given features.

---

## Feature Importance
The RandomForestRegressor provides insights into which features are most influential in predicting house prices. Key features include:
- **RM** (number of rooms): Strongly positively correlated with house prices.
- **LSTAT** (percentage of lower-status population): Strongly negatively correlated with house prices.
- **DIS** (distance to employment centers): Also plays a significant role.

A bar chart of feature importances is generated to visualize these insights.

---

## Saving and Loading the Model
The trained model is saved using `joblib` for easy deployment:
```python
joblib.dump(model, 'house_price_model.pkl')
```
To load the model later:
```python
model = joblib.load('house_price_model.pkl')
```

---

## User Input for Prediction
The project includes an interactive feature that allows users to input values for the 13 original features to predict the house price. The code calculates the `RM_TAX_RATIO` automatically and uses the trained model to make a prediction.

Example usage:
- Input values for crime rate, number of rooms, etc.
- The model predicts the house price based on the provided features.

---

## Conclusion and Future Work
This project successfully demonstrates the application of machine learning to predict house prices using the Boston Housing dataset. The RandomForestRegressor achieves strong performance, with an RMSE of $2,730 and an R² score of 0.90.

### Potential Improvements
- **Model Tuning**: Experiment with hyperparameter tuning (e.g., grid search) to further optimize the RandomForest model.
- **Alternative Models**: Try other regression algorithms like Gradient Boosting or Neural Networks.
- **Ethical Considerations**: Given the dataset's age and the inclusion of a potentially sensitive feature (`B`), consider using a more modern and ethically vetted dataset for future projects.

---

## License and Credits
- **Dataset Source**: The Boston Housing dataset is originally from the [UCI Machine Learning Repository](http://lib.stat.cmu.edu/datasets/boston).
- **License**: This project is licensed under the MIT License.
