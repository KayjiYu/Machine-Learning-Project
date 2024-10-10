## ---------------Import modules------------------
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

X = df.drop(['User_ID', 'Product_ID', 'Purchase'], axis=1)
y = df['Purchase']



from sklearn.metrics import mean_squared_error
## Model Training
from sklearn.model_selection import cross_val_score, train_test_split


def train(model, X, y):
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
    model.fit(X_train, y_train)

    # predict the results
    pred = model.predict(X_test)

    # cross validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("Results")
    print("MSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("CV Score:", np.sqrt(cv_score))

# LinearRegression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title='Model Coefficients')

# DecisionTree
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
train(model, X, y)
features = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='Feature Importance')

# RandomForest
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1)
train(model, X, y)
features = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='Feature Importance')

# Extra Tree
import time

start_time = time.time()

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor(n_jobs=-1)
train(model, X, y)
features = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
features.plot(kind='bar', title='Feature Importance')
print("-------------------------------")
print(f"Execution Time: {(time.time() - start_time):.2f} seconds")
