## ---------------Import modules------------------
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline
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










# type: ignore # f2.canvas.manager.window.wm_geometry('+2000+1000')

