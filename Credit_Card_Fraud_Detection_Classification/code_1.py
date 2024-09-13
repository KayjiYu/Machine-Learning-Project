# ----------- Import Modules-------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------- Load the dataset -----------

df = pd.read_csv("creditcard.csv")
df.head()

# statistical info
df.describe()

# datatype info
df.info()

# ------------Preprocessing data--------------

df.isnull().sum()

# ------------Exploratory Data Analysis---------

sns.countplot(df['Class'])

df_temp = df.drop(['Time', 'Amount', 'Class'], axis=1)

fig, ax = plt.subplots(ncols=4, nrows=7, figsize=(20,50))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.distplot(df_temp[col], ax=ax[index])
    index += 1

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=5)

sns.distplot(df['Time'])

sns.displot(df['Amount'])

sns.histplot(df['Class'])


# -------------Correlation Matrix-------------

corr = df.corr()
plt.figure(figsize=(30,40))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# ------------Input Split----------------

X = df.drop('Class', axis=1)
y = df['Class']


# -----------Standard Scaling----------------

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaler = sc.fit_transform(X)

X_scaler[-1]


# -----------Build Model--------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.25, random_state=42, stratify=y)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_lr_pred = model_lr.predict(X_test)
print(classification_report(y_test, y_lr_pred))
print(f"F1 Score: {f1_score(y_test, y_lr_pred)}")

# RandomForest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
y_rf_pred = model_rf.predict(X_test)
print(classification_report(y_test, y_rf_pred))
print(f"F1 Score: {f1_score(y_test, y_rf_pred)}")

# XGBoost
from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
y_xgb_pred = model_xgb.predict(X_test)
print(classification_report(y_test, y_xgb_pred))
print(f"F1 score: {f1_score(y_test, y_xgb_pred)}")


# ------------- Class Imbalancement------------
from imblearn.over_sampling import SMOTE
over_sample = SMOTE()
X_smote, y_smote = over_sample.fit_resample(X_train, y_train)

def train_model(model):
    model.fit(X_smote, y_smote)
    y_pred = model.predict(X_test)
    print(f"--------------{model}---------------")
    print(classification_report(y_test, y_pred))
    print(f"F1 score: {f1_score(y_test, y_pred)}")
    
models = [RandomForestClassifier(), 
          LogisticRegression(), 
          XGBClassifier()]
for model in models:
    train_model(model)