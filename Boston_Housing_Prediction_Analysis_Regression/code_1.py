## ------------Import modules--------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("Boston Dataset.csv")
df.head()
df.drop(columns=['Unnamed: 0'], axis=0, inplace=True)
df.head()


