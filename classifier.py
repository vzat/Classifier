import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Read training set
columnHeadings=['id', 'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome',
                'y']

trainingset = pd.read_csv('trainingset.txt', header=None, names=columnHeadings, index_col=False, na_values=['unknown'])

# Extract numeric features
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
numeric_dfs = trainingset[numeric_features]

# Extract categorical features
cat_dfs = trainingset.drop(numeric_features, axis=1)

c_cat_dfs = pd.get_dummies(cat_dfs)

# Create manual python dictonary

# # Convert categorical data to dictionaries
# cat_dfs = cat_dfs.T.to_dict().values()
#
# # Convert categorical values to numeric which are needed for sklearn
# # vecObj = DictVectorizer(sparse=False)
# le = LabelEncoder()
# vec_cat_dfs = le.fit_transform(cat_dfs)
#
# print(vec_cat_dfs)

# Merge numeric and categorical features into one set
# train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))

