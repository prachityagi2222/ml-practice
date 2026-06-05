"""data preparation"""
import pandas as pd
import numpy as np


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'

df = pd.read_csv(data)

df.columns = df.columns.str.lower().str.replace(' ', '_')
df['make'] = df['make'].str.lower().str.replace(' ', "_")

print(df.dtypes) #to print the types of the car
strings = list(df.dtypes[df.dtypes == 'object'].index)

"""data exploration analysis"""

import matplotlib.pyplot as plt
import seaborn as sns

# write in notebook so that all these plots are displayed in the notebook %matplotlib inline 
sns.histplot(df.msrp, bins=50)
plt.show()



