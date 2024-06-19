# Job Change Prediction

This project predicts the probability of a candidate looking for a new job or staying with their current company using a dataset from Kaggle. The dataset includes various features related to the candidate's background and current employment status.

## Dataset Features

- **enrollee_id**: Unique ID for the candidate.
- **city**: City code.
- **city_development_index**: Development index of the city (scaled).
- **gender**: Gender of the candidate.
- **relevent_experience**: Relevant experience of the candidate.
- **enrolled_university**: Type of university course enrolled in, if any.
- **education_level**: Education level of the candidate.
- **major_discipline**: Education major discipline of the candidate.
- **experience**: Candidate's total experience in years.
- **company_size**: Number of employees in the current employer's company.
- **company_type**: Type of current employer.
- **last_new_job**: Difference in years between the previous job and the current job.
- **training_hours**: Training hours completed.
- **target**: 0 – Not looking for a job change, 1 – Looking for a job change.

## Data Preprocessing and Visualization

### Handling Missing Values
- Replace null values with the most common value, 'Unknown', or using forward fill (ffill).

### Encoding Categorical Variables
- Use Label Encoding for ordinal data.
- Use One-Hot Encoding for nominal data.

### Handling Imbalanced Data
- Use SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset:
  ```python
  smote = SMOTE(sampling_strategy='minority')
  X, y = smote.fit_resample(X, y)
  ```
  
### Data Visualization
- Plot the distribution of the target variable.
- Plot the correlation matrix of the features.
- Plot the distribution of the numerical features.
- Plot the count of the categorical features.


## Model Building and Evaluation

### Model Selection
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting - XGBoost
- Gradient Boosting - LightGBM
- Decision Tree Classifier

### Model Evaluation
- Split the dataset into training and testing sets.
- Cross-validation to evaluate the models.
