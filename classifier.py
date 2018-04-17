import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read training set
columnHeadings=['id', 'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome',
                'y']

trainingset = pd.read_csv('trainingset.txt', header=None, names=columnHeadings, index_col=False, na_values=['unknown'])

# Extract target feature
targetLabels = trainingset['y']

# Extract numeric features
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
numeric_dfs = trainingset[numeric_features]

# Extract categorical features
# ID is removed as it is a unique value which will not help the classification
# The target feature is removed as well
cat_dfs = trainingset.drop(numeric_features + ['id'] + ['y'], axis=1)

# Fill missing values with unknown
cat_dfs.fillna('unknown', inplace=True)

# Replace missing values if they occur less than 10% of the time
for column in cat_dfs:
    job_freq = cat_dfs[column].value_counts(ascending=False)
    job_freq_dict = job_freq.to_dict()
    if 'unknown' in job_freq_dict:
        no_unknown_values = job_freq_dict['unknown']/len(cat_dfs[column]) * 100.0

        # Impute values that are missing less than 10 percent of values
        most_freq_key = job_freq.keys()[0]
        if most_freq_key != 'unknown' and no_unknown_values < 10:
            cat_dfs[column].replace(to_replace='unknown', value=most_freq_key, inplace=True)

# Convert categorical data to dictionaries
cat_dfs = cat_dfs.T.to_dict().values()

# Convert categorical values to numeric which are needed for sklearn
vecObj = DictVectorizer(sparse=False)
vec_cat_dfs = vecObj.fit_transform(cat_dfs)

# Merge numeric and categorical features into one set
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))

# Create Decision Tree
# treeModel = DecisionTreeClassifier(criterion='entropy')
# treeModel.fit(train_dfs, targetLabels)

# Test decision tree
train_values, test_values, train_target, test_target = train_test_split(train_dfs, targetLabels, test_size=0.4)
treeModel = DecisionTreeClassifier(criterion='entropy')
treeModel.fit(train_values, train_target)
predictions = treeModel.predict(test_values)
print("Accuracy= " + str(accuracy_score(test_target, predictions, normalize=True)))



