# Energy Consumption Prediction using Linear Regression

## Project Overview
This project aims to build a machine learning model to predict building energy consumption based on various structural features and environmental factors. The goal is to demonstrate the process of data loading, preprocessing, model training, and evaluation using a linear regression approach.

## Dataset
The dataset used in this project is the [Energy Consumption Dataset for Linear Regression](https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression) from Kaggle. It contains information on:
- Building Type
- Square Footage
- Number of Occupants
- Appliances Used
- Average Temperature
- Day of Week
- Energy Consumption (Target Variable)

## Methodology
The project follows a standard machine learning workflow:
1.  **Data Loading and Initial Exploration**: Loading the dataset and performing initial checks (e.g., `.info()`, `.describe()`, checking for nulls and duplicates).
2.  **Data Preprocessing**: Handling categorical variables using one-hot encoding. Investigating potential outliers (though none were found in the 'Energy Consumption' column based on the IQR method).
3.  **Model Training**: Training a Linear Regression model on the preprocessed training data.
4.  **Model Evaluation**: Evaluating the model's performance on a separate test dataset using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).

## Key Findings
- Initial data exploration revealed no missing values or duplicates.
- Categorical features were successfully transformed using one-hot encoding.
- The Linear Regression model achieved exceptionally high evaluation scores (R2 = 1.00, MAE, MSE, RMSE close to 0) on the test dataset. This indicates a very strong linear relationship between the features and the target in this specific dataset, likely due to its synthetic nature as discussed in the analysis.

## How to Run the Code
1.  Clone this repository to your local machine or open the `.ipynb` file directly in Google Colab.
2.  Ensure you have the necessary libraries installed (pandas, numpy, scikit-learn, matplotlib, seaborn). If running in Colab, most are pre-installed.
3.  Download the dataset from the Kaggle link provided above.
4.  Run the cells in the notebook sequentially to perform data loading, preprocessing, model training, and evaluation.

## Future Improvements
- Explore other regression models (e.g., Ridge, Lasso, RandomForestRegressor).
- Perform more in-depth feature engineering.
- Investigate the data distribution further if applying to a real-world scenario.
- Consider deployment options to make the model accessible via an API.

## Author
Rabjyotsingh Majethiya
(https://www.linkedin.com/in/rabjyotsingh/)

## License
This project is licensed under the MIT License.
