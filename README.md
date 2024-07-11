# AIDS Classification
 This notebook is designed to classify AIDS-related data using various machine learning algorithms. It includes data preprocessing, visualization, and model evaluation.

## Introduction
- This notebook focuses on the classification of AIDS-related data. It leverages multiple machine learning algorithms to predict the infection status based on the provided dataset.

## Libraries and Data Loading
- The following libraries are used in this notebook:

   pandas: For data manipulation and analysis.<br>
   numpy: For numerical computations.<br>
   matplotlib and seaborn: For data visualization.<br>
   plotly: For interactive visualizations.<br>
   scikit-learn: For machine learning algorithms and evaluation metrics.<br>
   xgboost: For the XGBoost classifier.<br>
   imbalanced-learn: For handling imbalanced datasets.<br>
   The data is loaded from a CSV file AIDS_Classification.csv.<br>

```python
#import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
```
```python
# Load data and show it
df = pd.read_csv("Data set/AIDS_Classification.csv")
df.head()
```
## Data Exploration
Initial data exploration includes:

- Displaying the dataset to understand its structure.
- Checking data information to understand the data types and identify missing values.
- Generating summary statistics to understand the distribution of data.
```python
# Show information about data
df.info()
```
```python
# Show distribution about data
df.describe()
```
## Data Visualization
Visualizing data distributions and correlations to understand relationships and identify patterns.

``` python
# Show correlation between target and columns
df.corr()['infected'].sort_values()
```

![corrlation ADIS](https://github.com/Mahmedorabi/ADIS-Classifcation/assets/105740465/16fa7b14-ba9a-4528-9e65-e438a6a9fd59)

Distribution about data

![Dist ADIS](https://github.com/Mahmedorabi/ADIS-Classifcation/assets/105740465/953c3620-fb90-4584-83dd-93c530eac933)

## Data Preprocessing
Preprocessing steps involve handling imbalanced data using oversampling techniques.


![Unblance ADIS](https://github.com/Mahmedorabi/ADIS-Classifcation/assets/105740465/c39d6b56-34e7-43a4-8e24-5bc10f59749e)

```python

# fixed unbalance classes using Over sampling
sampler=RandomOverSampler(random_state=42)

x_sample,y_sample=sampler.fit_resample(x,y)
```
## Model Training and Evaluation
Several machine learning models are trained and evaluated, including:

- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- Support Vector Classifier (SVC)
Each model is trained using a portion of the dataset, and various evaluation metrics are used to assess their performance.

```python
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Example: Training a Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
```
## Evaluation of our model
<div>
   <img src="Grid ADIS.png" alt="Grid"/>
</div>

```
Accuracy of DecisionTree is : 84.41358024691358%
--------------------------------------------
classification report of DecisionTree is 
               precision    recall  f1-score   support

           0       0.87      0.82      0.85       336
           1       0.82      0.87      0.84       312

    accuracy                           0.84       648
   macro avg       0.84      0.85      0.84       648
weighted avg       0.85      0.84      0.84       648
```
```
Accuracy of Random Forest is : 92.74691358024691%
--------------------------------------------
classification report of Random Forest is 
               precision    recall  f1-score   support

           0       0.95      0.91      0.93       336
           1       0.91      0.95      0.93       312

    accuracy                           0.93       648
   macro avg       0.93      0.93      0.93       648
weighted avg       0.93      0.93      0.93       648

```
```

Accuracy of Ada Boost is : 84.41358024691358%
--------------------------------------------
classification report of Ada Boost is 
               precision    recall  f1-score   support

           0       0.87      0.82      0.85       336
           1       0.82      0.87      0.84       312

    accuracy                           0.84       648
   macro avg       0.84      0.85      0.84       648
weighted avg       0.85      0.84      0.84       648
```
```

Accuracy of xgboosting is : 95.67901234567901%
--------------------------------------------
classification report of xgboosting is 
               precision    recall  f1-score   support

           0       0.98      0.94      0.96       336
           1       0.94      0.97      0.96       312

    accuracy                           0.96       648
   macro avg       0.96      0.96      0.96       648
weighted avg       0.96      0.96      0.96       648
```
```
Accuracy of SVC is : 84.87654320987654%
--------------------------------------------
classification report of SVC is 
               precision    recall  f1-score   support

           0       0.86      0.85      0.85       336
           1       0.84      0.85      0.84       312

    accuracy                           0.85       648
   macro avg       0.85      0.85      0.85       648
weighted avg       0.85      0.85      0.85       648
```
```
Accuracy of GradientBoosting is : 94.9074074074074%
--------------------------------------------
classification report of GradientBoosting is 
               precision    recall  f1-score   support

           0       0.98      0.92      0.95       336
           1       0.92      0.98      0.95       312

    accuracy                           0.95       648
   macro avg       0.95      0.95      0.95       648
weighted avg       0.95      0.95      0.95       648
```


## Conclusion
The notebook concludes with a summary of the model performances and insights gained from the analysis. It provides recommendations for future work and potential improvements.

