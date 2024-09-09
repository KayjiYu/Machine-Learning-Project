## ------------Import modules--------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("Boston_Dataset.csv")
df.head()
df.drop(columns=['Unnamed: 0'], axis=0, inplace=True)
df.head()

# statistical info
df.describe()

# datatype info
df.info


## ----------Preprocessing the dataset---------

# check for null values
df.isnull().sum()
