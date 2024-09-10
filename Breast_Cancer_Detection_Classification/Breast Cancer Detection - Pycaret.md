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

## Install Pycaret Module

```python
# !pip install pycaret
```

## Import Modules

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pycaret.classification import *
%matplotlib inline
warnings.filterwarnings('ignore')
```

## Load the Dataset

```python
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
```

```python
# delete unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'], axis=1)
```

```python
# statistical info
df.describe()
```

```python
# datatype info
df.info()
```

## Exploratory Data Analysis

```python
sns.countplot(df['diagnosis'])
```

```python
df_temp = df.drop(columns=['diagnosis'], axis=1)
```

```python
# create dist plot
fig, ax = plt.subplots(ncols=6, nrows=5, figsize=(20, 20))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.distplot(df[col], ax=ax[index])
    index+=1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
```

```python
# create box plot
fig, ax = plt.subplots(ncols=6, nrows=5, figsize=(20, 20))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.boxplot(y=col, data=df, ax=ax[index])
    index+=1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
```

## Create and Train the Model

```python
# setup the data
clf = setup(df, target='diagnosis')
```

```python
# train and test the models
compare_models()
```

```python
# select the best model
model = create_model('catboost')
```

```python
# hyperparameter tuning
best_model = tune_model(model)
```

```python
evaluate_model(best_model)
```

```python
# plot the results
plot_model(estimator=best_model, plot='confusion_matrix')
```

```python

```
