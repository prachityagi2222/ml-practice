import pandas as pd
import numpy as np

"""data preparation"""
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'

df = pd.read_csv(data) #convert it into data frame
df.columns = df.columns.str.lower().str.replace(' ', '_') #replacing the spaces with underscore, and covnerting them in lower case
strings = list(df.dtypes[df.dtypes == 'object'].index)

for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_') #replacing the values in the columns


"""exploratory data analysis: Looking at the data from every angle before touching the ML model
1. look at data (no of cols/row, what does sample look like)
2. check data types
3. check missing values
4. understand target variable(is it skewed, does it have outliers)
5.  check categorial features (how many brands, how many categories)
6. find unusual values.
7. understand relationships (does engine size affect price? does age affect price? which brands are expensive?)
"""

for col in df.columns:
    # print(col)
    # print(df[col].unique()[:5])
    # print(df[col].nunique())
    # print()
    pass

#distribution of price
import matplotlib.pyplot as plt
import seaborn as sns

# sns.histplot(df.msrp, bins=50)
# plt.show()

# sns.histplot(df.msrp[df.msrp < 100000], bins=50)
# plt.show()


np.log1p([0, 1, 10, 1000, 100000]) #doing log transformation, as it compresses large numbers by a lot, adn small numbers by a little
price_logs = np.log1p(df.msrp)
# sns.histplot(price_logs, bins=50)

df.isnull().sum() #if some value is null, NaN, then it replaces it with true, else with false. .sum() calculates the sum


"""setting up validation framework
1. splitting the dataframework
2. seprating features and targets
3. resetting indices
4. making sure the split is reproducible
"""
n = len(df)
n_val = int(n * 0.2)
n_test = int(n*0.2)
n_train = n - n_val - n_test

df.iloc[[10, 0, 3, 5]]

df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train+n_val]
df_test = df.iloc[n_train+n_val:]

idx = np.arange(n)

np.random.seed(2)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val:]]
df_test = df.iloc[idx[n_train+n_val:]]

df_train = df_train.reset_index(drop=True) #drop=True, old index is discarded, the new index is kept
df_val = df_val.reset_index(drop=True) #reset_index, assigns the new index
df_test = df_test.reset_index(drop=True)

#saving the values to be predicted in y
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

#deleteing the column we need to predict
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

"""linear regression: just taking an example
"""

xi = [453, 11, 86]
w0 = 7.17
w = [0.01, 0.04, 0.002]

def linear_regression(xi):
    n = len(xi)
    pred = w0
    for j in range(n):
        pred = pred + w[j]* xi[j]

    return pred

# np.log1p(np.exmp1(linear_regression(xi)))

"""2.6 linear regression, vector form"""

# created a new dot function
# xi =  [1] + xi
# w_new = [w0] + w
def dot(xi, w):
    n = len(xi)
    
    res = 0.0
    
    for j in range(n):
        res = res + xi[j] * w[j]
    
    return res

def linear_regression(xi):
    return w0 + dot(xi, w)

w_new = [w0] + w

def linear_regression(xi):
    xi = [1] + xi
    return dot(xi, w_new)

w0 = 7.17
w = [0.01, 0.04, 0.002]
w_new = [w0] + w

def linear_regression(X):
    return X.dot(w_new)


"""training linear regression model"""
X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38,  54, 185],
    [142, 25, 431],
    [453, 31, 86],
]

X = np.array(X)
ones = np.ones(X.shape[0])
X = np.column_stack([ones, X]) #making the new xi, each one of which has 1 before

y = [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000]

XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
w_full = XTX_inv.dot(X.T).dot(y)

# this function will train the linear regression, that is find out the value of w that fit the training data
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


"""car price baseline model

1. Create X (feature matrix)
2. Train linear regression model (find w0 and w)
3. Predict prices for the training set
4. Compare predicted and actual distributions
"""

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
X_train = df_train[base].fillna(0).values
w0, w = train_linear_regression(X_train, y_train)
y_pred = w0 + X_train.dot(w)

sns.histplot(y_pred, color='blue', alpha=0.5, bins=50)
sns.histplot(y_train, color='red', alpha=0.5, bins=50)
plt.show()

"""rmse"""

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

"""validating the model"""
def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

"""simple feature engineering"""
def prepare_X(df):
    df = df.copy()
    
    df['age'] = 2017 - df['year']
    features = base + ['age']
    
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

sns.histplot(y_pred, label='prediction', color='red', alpha=0.5, bins=50)
sns.histplot(y_val, label='target', color='blue',  alpha=0.5, bins=50)
plt.legend()
plt.show()

"""categorial variables"""
categorical_columns = [
    'make', 'model', 'engine_fuel_type', 'driven_wheels', 'market_category',
    'vehicle_size', 'vehicle_style']

categorical = {}

for c in categorical_columns:
    categorical[c] = list(df_train[c].value_counts().head().index)

def prepare_X(df):
    df = df.copy()

    df['age'] = 2017 - df['year']
    features = base + df['age'] 

    for v in [2, 3, 4]:
        df['num_doors_%d' ]

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

"""regularisation"""
X = [
    [4, 4, 4],
    [3, 5, 5],
    [5, 1, 1],
    [5, 4, 4],
    [7, 5, 5],
    [4, 5, 5.00000001],
]

X = np.array(X)

XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
XTX_inv.dot(X.T).dot(y)


array([[ -0.33668908,   0.33501399,   0.33501399],
       [  0.33501399,  49.91590897, -50.08509104],
       [  0.33501399, -50.08509104,  49.91590897]])

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)
