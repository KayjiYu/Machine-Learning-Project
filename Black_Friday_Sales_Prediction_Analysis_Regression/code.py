## ---------------Import modules------------------
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib 
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


## ------------Loading the dataset----------------
df = pd.read_csv("train.csv")
df.head()

# statistical info
df.describe()

# datatype info
df.info()

# find unique values
df.apply(lambda x: len(x.unique()))


## ---------Exploratory Data Analysis-------------

# def plot_geometry(fig, x, y):
#     return fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")

sns.displot(df["Purchase"], bins=25)
# f0.canvas.manager.window.wm_geometry('+2000+0')


# distribution of numeric variables
sns.countplot(df["Gender"])
# f1.canvas.manager.window.wm_geometry('+2000+500')

sns.countplot(df['Age'])



## ---------Preprocessing the dataset---------

# check for null values
df.isnull().sum()

df['Product_Category_2'] = df['Product_Category_2'].fillna(-2.0).astype("float32")
df['Product_Category_3'] = df['Product_Category_3'].fillna(-2.0).astype("float32")

df.isnull().sum()

# encoding vales using dict
gender_dict = {'F':0, 'M':1}
df['Gender'] = df['Gender'].apply(lambda x: gender_dict[x])
df.head()

# to improve the metric use one hot encoding
# label encoding
cols = ['Age', 'City_Category', 'Stay_In_Current_City_Years']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()



## ----------Input Split------------------
corr = df.drop(['User_ID', 'Product_ID'], axis=1).corr()
plt.figure(figsize=(14, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm')



## ----------Input Split------------------
df.head()

X = df.drop(['User_ID', 'Product_ID', 'Purchase'])
y = df['Purchase']










# type: ignore # f2.canvas.manager.window.wm_geometry('+2000+1000')

