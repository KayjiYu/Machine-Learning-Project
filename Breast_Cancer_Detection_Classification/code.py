## --------------- Import Modules------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pycaret import classification

%matplotlib

## --------------Load the Dataset------------------
df = pd.read_csv('data.csv')
df.head()

# delete unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'], axis=1)

# statistical info
df.describe()

# datatype info
df.info()


## --------------Exploratory Data Analysis----------------
sns.countplot(df['diagnosis'])

df_temp = df.drop(columns=['diagnosis'], axis=1)
fig, ax = plt.subplots(ncols=6, nrows=5, figsize=(20, 20))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.distplot(df[col], ax=ax[index])
    index+=1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# create box plot
fig, ax = plt.subplots(ncols=6, nrows=5, figsize=(20, 20))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.boxplot(y=col, data=df, ax=ax[index])
    index+=1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
