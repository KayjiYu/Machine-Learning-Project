# -----------Import Modules------------

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


# -------- Loading the dataset---------

df = pd.read_csv("Train.csv")
df.head()

# statistical info
df.describe()
df.info()

# check unique values in dataset
df.apply(lambda x: len(x.unique()))


# ----------Preprocessing the dataset------

df.isnull().sum()

# check for categorical attributes
cat_col = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        cat_col.append(x)
cat_col

cat_col.remove("Item_Identifier")
cat_col.remove("Outlet_Identifier")
cat_col


# print the categorical columns
for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()

# fill the missing values
item_weight_mean = df.pivot_table(values = "Item_Weight",
                                  index = "Item_Identifier")
item_weight_mean

miss_bool = df['Item_Weight'].isnull()
miss_bool

for i, item in enumerate(df['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean:
            df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            df['Item_Weight'][i] = np.mean(df['Item_Weight'])

df['Item_Weight'].isnull().sum()

outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
outlet_size_mode

miss_bool = df['Outlet_Size'].isnull()
df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

df['Outlet_Size'].isnull().sum()

sum(df['Item_Visibility'] == 0)

# replace zeros with mean
df.loc[:, 'Item_Visibility'].replace([0], [df['Item_Visibility'].mean()], inplace=True)
sum(df['Item_Visibility']==0)

# combine item fat content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
df['Item_Fat_Content'].value_counts()


# -------------Creation of New Attributes--------------

df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
df['New_Item_Type']

df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
df['New_Item_Type'].value_counts()

df.loc[df['New_Item_Type'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
df['Item_Fat_Content'].value_counts()

# create small values for establishment year
df['Outlet_Years'] = 2003 - df['Outlet_Establishment_Year']
df['Outlet_Years']



# -----------------Exploratory Data Analysis-------------
df.head()

sns.displot(df['Item_Weight'])

sns.histplot(df['Item_Visibility'])

sns.countplot(df["Item_Fat_Content"])

label = list(df['Item_Type'].unique())
chart = sns.countplot(df["Item_Type"])
chart.set_xticklabels(labels=label, rotation=90)

sns.countplot(df['Outlet_Establishment_Year'])

sns.countplot(df['Outlet_Size'])

# -----------------Coorelation Matrix--------------------
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')


# -----------------Label Encoding--------------------
from sklearn.metrics.pairwise import normalize
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
cat_col = ['Item_Fat_Content', "Item_Type", 'Outlet_Size', 
           'Outlet_Location_Type', "Outlet_Type", "New_Item_Type"]
for col in cat_col:
    df[col] = le.fit_transform(df[col])

# -----------------Onehot Encoding------------------
df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Outlet_Size',
                                  'Outlet_Location_Type', 'Outlet_Type',
                                  "New_Item_Type"])
df.head()

# ------------------Input Split---------------------
X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier',
                     'Outlet_Identifier', 'Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']


from sklearn.metrics import mean_squared_error
# ----------------Model Training--------------------
from sklearn.model_selection import cross_val_score


def train(model, X, y):
    # train th model
    model.fit(X, y)
    # predict the training set
    pred = model.predict(X)
    # perform cross-validation
    cv_score = cross_val_score(model, X, y,
                               scoring='neg_mean_squared_error',
                               cv=5)
    cv_score = np.abs(np.mean(cv_score))

    print("Model Report")
    print("MSE", mean_squared_error(y, pred))
    print("CV Score:", cv_score)

from sklearn.linear_model import Lasso, LinearRegression, Ridge

model = LinearRegression()
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")


model = Lasso()
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")
