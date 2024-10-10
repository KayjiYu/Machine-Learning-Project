---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.054999, "end_time": "2021-10-26T14:39:51.732722", "exception": false, "start_time": "2021-10-26T14:39:51.677723", "status": "completed"}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

<!-- #region papermill={"duration": 0.0372, "end_time": "2021-10-26T14:39:51.809118", "exception": false, "start_time": "2021-10-26T14:39:51.771918", "status": "completed"} -->
## Import Modules
<!-- #endregion -->

```python papermill={"duration": 0.872959, "end_time": "2021-10-26T14:39:52.719543", "exception": false, "start_time": "2021-10-26T14:39:51.846584", "status": "completed"}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```

<!-- #region papermill={"duration": 0.037974, "end_time": "2021-10-26T14:39:52.801957", "exception": false, "start_time": "2021-10-26T14:39:52.763983", "status": "completed"} -->
## Loading the dataset
<!-- #endregion -->

```python papermill={"duration": 0.090114, "end_time": "2021-10-26T14:39:52.930465", "exception": false, "start_time": "2021-10-26T14:39:52.840351", "status": "completed"}
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
```

```python papermill={"duration": 0.080848, "end_time": "2021-10-26T14:39:53.110020", "exception": false, "start_time": "2021-10-26T14:39:53.029172", "status": "completed"}
## statistical info
train.describe()
```

```python papermill={"duration": 0.056979, "end_time": "2021-10-26T14:39:53.216348", "exception": false, "start_time": "2021-10-26T14:39:53.159369", "status": "completed"}
## datatype info
train.info()
```

<!-- #region papermill={"duration": 0.038578, "end_time": "2021-10-26T14:39:53.296624", "exception": false, "start_time": "2021-10-26T14:39:53.258046", "status": "completed"} -->
## Exploratory Data Analysis
<!-- #endregion -->

```python papermill={"duration": 0.240071, "end_time": "2021-10-26T14:39:53.575827", "exception": false, "start_time": "2021-10-26T14:39:53.335756", "status": "completed"}
## categorical attributes
sns.countplot(train['Survived'])
```

```python papermill={"duration": 0.222447, "end_time": "2021-10-26T14:39:53.843428", "exception": false, "start_time": "2021-10-26T14:39:53.620981", "status": "completed"}
sns.countplot(train['Pclass'])
```

```python papermill={"duration": 0.315432, "end_time": "2021-10-26T14:39:54.203593", "exception": false, "start_time": "2021-10-26T14:39:53.888161", "status": "completed"}
sns.countplot(train['Sex'])
```

```python papermill={"duration": 0.250733, "end_time": "2021-10-26T14:39:54.495967", "exception": false, "start_time": "2021-10-26T14:39:54.245234", "status": "completed"}
sns.countplot(train['SibSp'])
```

```python papermill={"duration": 0.264688, "end_time": "2021-10-26T14:39:54.803265", "exception": false, "start_time": "2021-10-26T14:39:54.538577", "status": "completed"}
sns.countplot(train['Parch'])
```

```python papermill={"duration": 0.221603, "end_time": "2021-10-26T14:39:55.077587", "exception": false, "start_time": "2021-10-26T14:39:54.855984", "status": "completed"}
sns.countplot(train['Embarked'])
```

```python papermill={"duration": 0.311216, "end_time": "2021-10-26T14:39:55.432668", "exception": false, "start_time": "2021-10-26T14:39:55.121452", "status": "completed"}
## numerical attributes
sns.distplot(train['Age'])
```

```python papermill={"duration": 0.354661, "end_time": "2021-10-26T14:39:55.834852", "exception": false, "start_time": "2021-10-26T14:39:55.480191", "status": "completed"}
sns.distplot(train['Fare'])
```

```python papermill={"duration": 0.260207, "end_time": "2021-10-26T14:39:56.140666", "exception": false, "start_time": "2021-10-26T14:39:55.880459", "status": "completed"}
class_fare = train.pivot_table(index='Pclass', values='Fare')
class_fare.plot(kind='bar')
plt.xlabel('Pclass')
plt.ylabel('Avg. Fare')
plt.xticks(rotation=0)
plt.show()
```

```python papermill={"duration": 0.244072, "end_time": "2021-10-26T14:39:56.431124", "exception": false, "start_time": "2021-10-26T14:39:56.187052", "status": "completed"}
class_fare = train.pivot_table(index='Pclass', values='Fare', aggfunc=np.sum)
class_fare.plot(kind='bar')
plt.xlabel('Pclass')
plt.ylabel('Total Fare')
plt.xticks(rotation=0)
plt.show()
```

```python papermill={"duration": 0.407463, "end_time": "2021-10-26T14:39:56.887414", "exception": false, "start_time": "2021-10-26T14:39:56.479951", "status": "completed"}
sns.barplot(data=train, x='Pclass', y='Fare', hue='Survived')
```

```python papermill={"duration": 0.407863, "end_time": "2021-10-26T14:39:57.343064", "exception": false, "start_time": "2021-10-26T14:39:56.935201", "status": "completed"}
sns.barplot(data=train, x='Survived', y='Fare', hue='Pclass')
```

<!-- #region papermill={"duration": 0.048157, "end_time": "2021-10-26T14:39:57.439959", "exception": false, "start_time": "2021-10-26T14:39:57.391802", "status": "completed"} -->
## Data Preprocessing
<!-- #endregion -->

```python papermill={"duration": 0.074167, "end_time": "2021-10-26T14:39:57.563338", "exception": false, "start_time": "2021-10-26T14:39:57.489171", "status": "completed"}
train_len = len(train)
# combine two dataframes
df = pd.concat([train, test], axis=0)
df = df.reset_index(drop=True)
df.head()
```

```python papermill={"duration": 0.065557, "end_time": "2021-10-26T14:39:57.679841", "exception": false, "start_time": "2021-10-26T14:39:57.614284", "status": "completed"}
df.tail()
```

```python papermill={"duration": 0.059742, "end_time": "2021-10-26T14:39:57.788872", "exception": false, "start_time": "2021-10-26T14:39:57.729130", "status": "completed"}
## find the null values
df.isnull().sum()
```

```python papermill={"duration": 0.057113, "end_time": "2021-10-26T14:39:57.895471", "exception": false, "start_time": "2021-10-26T14:39:57.838358", "status": "completed"}
# drop or delete the column
df = df.drop(columns=['Cabin'], axis=1)
```

```python papermill={"duration": 0.058866, "end_time": "2021-10-26T14:39:58.003680", "exception": false, "start_time": "2021-10-26T14:39:57.944814", "status": "completed"}
df['Age'].mean()
```

```python papermill={"duration": 0.058801, "end_time": "2021-10-26T14:39:58.113135", "exception": false, "start_time": "2021-10-26T14:39:58.054334", "status": "completed"}
# fill missing values using mean of the numerical column
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
```

```python papermill={"duration": 0.05757, "end_time": "2021-10-26T14:39:58.220901", "exception": false, "start_time": "2021-10-26T14:39:58.163331", "status": "completed"}
df['Embarked'].mode()[0]
```

```python papermill={"duration": 0.058328, "end_time": "2021-10-26T14:39:58.329215", "exception": false, "start_time": "2021-10-26T14:39:58.270887", "status": "completed"}
# fill missing values using mode of the categorical column
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

<!-- #region papermill={"duration": 0.049573, "end_time": "2021-10-26T14:39:58.428350", "exception": false, "start_time": "2021-10-26T14:39:58.378777", "status": "completed"} -->
## Log transformation for uniform data distribution
<!-- #endregion -->

```python papermill={"duration": 0.367478, "end_time": "2021-10-26T14:39:58.845612", "exception": false, "start_time": "2021-10-26T14:39:58.478134", "status": "completed"}
sns.distplot(df['Fare'])
```

```python papermill={"duration": 0.058977, "end_time": "2021-10-26T14:39:58.956180", "exception": false, "start_time": "2021-10-26T14:39:58.897203", "status": "completed"}
df['Fare'] = np.log(df['Fare']+1)
```

```python papermill={"duration": 0.336811, "end_time": "2021-10-26T14:39:59.347631", "exception": false, "start_time": "2021-10-26T14:39:59.010820", "status": "completed"}
sns.distplot(df['Fare'])
```

<!-- #region papermill={"duration": 0.051724, "end_time": "2021-10-26T14:39:59.451308", "exception": false, "start_time": "2021-10-26T14:39:59.399584", "status": "completed"} -->
## Correlation Matrix
<!-- #endregion -->

```python papermill={"duration": 0.631059, "end_time": "2021-10-26T14:40:00.135031", "exception": false, "start_time": "2021-10-26T14:39:59.503972", "status": "completed"}
corr = df.corr()
plt.figure(figsize=(15, 9))
sns.heatmap(corr, annot=True, cmap='coolwarm')
```

```python papermill={"duration": 0.073753, "end_time": "2021-10-26T14:40:00.262390", "exception": false, "start_time": "2021-10-26T14:40:00.188637", "status": "completed"}
df.head()
```

```python papermill={"duration": 0.07386, "end_time": "2021-10-26T14:40:00.390607", "exception": false, "start_time": "2021-10-26T14:40:00.316747", "status": "completed"}
## drop unnecessary columns
df = df.drop(columns=['Name', 'Ticket'], axis=1)
df.head()
```

<!-- #region papermill={"duration": 0.054625, "end_time": "2021-10-26T14:40:00.500134", "exception": false, "start_time": "2021-10-26T14:40:00.445509", "status": "completed"} -->
## Label Encoding
<!-- #endregion -->

```python papermill={"duration": 0.182109, "end_time": "2021-10-26T14:40:00.738396", "exception": false, "start_time": "2021-10-26T14:40:00.556287", "status": "completed"}
from sklearn.preprocessing import LabelEncoder
cols = ['Sex', 'Embarked']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()
```

<!-- #region papermill={"duration": 0.056568, "end_time": "2021-10-26T14:40:00.850393", "exception": false, "start_time": "2021-10-26T14:40:00.793825", "status": "completed"} -->
## Train-Test Split
<!-- #endregion -->

```python papermill={"duration": 0.062281, "end_time": "2021-10-26T14:40:00.967568", "exception": false, "start_time": "2021-10-26T14:40:00.905287", "status": "completed"}
train = df.iloc[:train_len, :]
test = df.iloc[train_len:, :]
```

```python papermill={"duration": 0.071803, "end_time": "2021-10-26T14:40:01.094826", "exception": false, "start_time": "2021-10-26T14:40:01.023023", "status": "completed"}
train.head()
```

```python papermill={"duration": 0.074994, "end_time": "2021-10-26T14:40:01.228184", "exception": false, "start_time": "2021-10-26T14:40:01.153190", "status": "completed"}
test.head()
```

```python papermill={"duration": 0.066526, "end_time": "2021-10-26T14:40:01.352355", "exception": false, "start_time": "2021-10-26T14:40:01.285829", "status": "completed"}
# input split
X = train.drop(columns=['PassengerId', 'Survived'], axis=1)
y = train['Survived']
```

```python papermill={"duration": 0.073156, "end_time": "2021-10-26T14:40:01.483008", "exception": false, "start_time": "2021-10-26T14:40:01.409852", "status": "completed"}
X.head()
```

<!-- #region papermill={"duration": 0.05648, "end_time": "2021-10-26T14:40:01.595542", "exception": false, "start_time": "2021-10-26T14:40:01.539062", "status": "completed"} -->
## Model Training
<!-- #endregion -->

```python papermill={"duration": 0.125748, "end_time": "2021-10-26T14:40:01.777684", "exception": false, "start_time": "2021-10-26T14:40:01.651936", "status": "completed"}
from sklearn.model_selection import train_test_split, cross_val_score
# classify column
def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))
    
    score = cross_val_score(model, X, y, cv=5)
    print('CV Score:', np.mean(score))
```

```python papermill={"duration": 0.43207, "end_time": "2021-10-26T14:40:02.272542", "exception": false, "start_time": "2021-10-26T14:40:01.840472", "status": "completed"}
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model)
```

```python papermill={"duration": 0.190053, "end_time": "2021-10-26T14:40:02.519031", "exception": false, "start_time": "2021-10-26T14:40:02.328978", "status": "completed"}
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model)
```

```python papermill={"duration": 1.191513, "end_time": "2021-10-26T14:40:03.768926", "exception": false, "start_time": "2021-10-26T14:40:02.577413", "status": "completed"}
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model)
```

```python papermill={"duration": 0.969994, "end_time": "2021-10-26T14:40:04.796414", "exception": false, "start_time": "2021-10-26T14:40:03.826420", "status": "completed"}
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
classify(model)
```

```python papermill={"duration": 0.658113, "end_time": "2021-10-26T14:40:05.512880", "exception": false, "start_time": "2021-10-26T14:40:04.854767", "status": "completed"}
from xgboost import XGBClassifier
model = XGBClassifier()
classify(model)
```

```python papermill={"duration": 1.79848, "end_time": "2021-10-26T14:40:07.370311", "exception": false, "start_time": "2021-10-26T14:40:05.571831", "status": "completed"}
from lightgbm import LGBMClassifier
model = LGBMClassifier()
classify(model)
```

```python papermill={"duration": 5.453449, "end_time": "2021-10-26T14:40:12.891210", "exception": false, "start_time": "2021-10-26T14:40:07.437761", "status": "completed"}
from catboost import CatBoostClassifier
model = CatBoostClassifier(verbose=0)
classify(model)
```

<!-- #region papermill={"duration": 0.060015, "end_time": "2021-10-26T14:40:13.011975", "exception": false, "start_time": "2021-10-26T14:40:12.951960", "status": "completed"} -->
## Complete Model Training with Full Data
<!-- #endregion -->

```python papermill={"duration": 0.15166, "end_time": "2021-10-26T14:40:13.223514", "exception": false, "start_time": "2021-10-26T14:40:13.071854", "status": "completed"}
model = LGBMClassifier()
model.fit(X, y)
```

```python papermill={"duration": 0.076399, "end_time": "2021-10-26T14:40:13.360082", "exception": false, "start_time": "2021-10-26T14:40:13.283683", "status": "completed"}
test.head()
```

```python papermill={"duration": 0.068878, "end_time": "2021-10-26T14:40:13.489471", "exception": false, "start_time": "2021-10-26T14:40:13.420593", "status": "completed"}
# input split for test data
X_test = test.drop(columns=['PassengerId', 'Survived'], axis=1)
```

```python papermill={"duration": 0.075487, "end_time": "2021-10-26T14:40:13.625263", "exception": false, "start_time": "2021-10-26T14:40:13.549776", "status": "completed"}
X_test.head()
```

```python papermill={"duration": 0.082263, "end_time": "2021-10-26T14:40:13.769079", "exception": false, "start_time": "2021-10-26T14:40:13.686816", "status": "completed"}
pred = model.predict(X_test)
pred
```

<!-- #region papermill={"duration": 0.061044, "end_time": "2021-10-26T14:40:13.893390", "exception": false, "start_time": "2021-10-26T14:40:13.832346", "status": "completed"} -->
## Test Submission
<!-- #endregion -->

```python papermill={"duration": 0.081641, "end_time": "2021-10-26T14:40:14.038101", "exception": false, "start_time": "2021-10-26T14:40:13.956460", "status": "completed"}
sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
sub.head()
```

```python papermill={"duration": 0.075318, "end_time": "2021-10-26T14:40:14.175740", "exception": false, "start_time": "2021-10-26T14:40:14.100422", "status": "completed"}
sub.info()
```

```python papermill={"duration": 0.070529, "end_time": "2021-10-26T14:40:14.308354", "exception": false, "start_time": "2021-10-26T14:40:14.237825", "status": "completed"}
sub['Survived'] = pred
sub['Survived'] = sub['Survived'].astype('int')
```

```python papermill={"duration": 0.076149, "end_time": "2021-10-26T14:40:14.447221", "exception": false, "start_time": "2021-10-26T14:40:14.371072", "status": "completed"}
sub.info()
```

```python papermill={"duration": 0.073952, "end_time": "2021-10-26T14:40:14.582840", "exception": false, "start_time": "2021-10-26T14:40:14.508888", "status": "completed"}
sub.head()
```

```python papermill={"duration": 0.073297, "end_time": "2021-10-26T14:40:14.719115", "exception": false, "start_time": "2021-10-26T14:40:14.645818", "status": "completed"}
sub.to_csv('submission.csv', index=False)
```

```python papermill={"duration": 0.062002, "end_time": "2021-10-26T14:40:14.844341", "exception": false, "start_time": "2021-10-26T14:40:14.782339", "status": "completed"}

```

```python papermill={"duration": 0.062958, "end_time": "2021-10-26T14:40:14.969741", "exception": false, "start_time": "2021-10-26T14:40:14.906783", "status": "completed"}

```

```python papermill={"duration": 0.061686, "end_time": "2021-10-26T14:40:15.093961", "exception": false, "start_time": "2021-10-26T14:40:15.032275", "status": "completed"}

```
