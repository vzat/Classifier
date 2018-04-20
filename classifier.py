import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv

pd.options.mode.chained_assignment = None

# Set headings
columnHeadings=['id', 'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome',
                'y']

# Read training set
trainingset = pd.read_csv('trainingset.txt', header=None, names=columnHeadings, index_col=False, na_values=['unknown'])

# Read queries
queries = pd.read_csv('queries.txt', header=None, names=columnHeadings, index_col=False, na_values=['unknown'])


# Extract target feature
targetLabels = trainingset['y']

# Extract numeric features
# Duration was removed as it had a cardinality of 1
numeric_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
numeric_dfs = trainingset[numeric_features]
numeric_queries_dfs = queries[numeric_features]


# Clamp the outlier on the balance
maxBalance = int(np.mean(numeric_dfs['balance']) + 2 * np.std(numeric_dfs['balance']))
numeric_dfs.loc[numeric_dfs['balance'] > maxBalance, 'balance'] = maxBalance


# Extract categorical features
# ID is removed as it is a unique value which will not help the classification
# The target feature is removed as well
cat_dfs = trainingset.drop(numeric_features + ['id'] + ['y'] + ['duration'], axis=1)
cat_queries_dfs = queries.drop(numeric_features + ['id'] + ['y'] + ['duration'], axis=1)


# Fill missing values with unknown
cat_dfs.fillna('unknown', inplace=True)
cat_queries_dfs.fillna('unknown', inplace=True)

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
            cat_queries_dfs[column].replace(to_replace='unknown', value=most_freq_key, inplace=True)


# Convert categorical data to dictionaries
cat_dfs = cat_dfs.T.to_dict().values()
cat_queries_dfs = cat_queries_dfs.T.to_dict().values()

# Convert categorical values to numeric which are needed for sklearn
vecObj = DictVectorizer(sparse=False)
vec_cat_dfs = vecObj.fit_transform(cat_dfs)
vec_cat_queries_dfs = vecObj.fit_transform(cat_queries_dfs)


# Merge numeric and categorical features into one set
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))
queries_dfs = np.hstack((numeric_queries_dfs.as_matrix(), vec_cat_queries_dfs))


# Test Decision Tree
train_values, test_values, train_target, test_target = train_test_split(train_dfs, targetLabels, test_size=0.4)
testDecisionTreeClassifier = DecisionTreeClassifier(criterion='entropy')
testDecisionTreeClassifier.fit(train_values, train_target)
predictions = testDecisionTreeClassifier.predict(test_values)
print("Decision Tree Accuracy = " + str(accuracy_score(test_target, predictions, normalize=True)))

# Test KNN
train_values, test_values, train_target, test_target = train_test_split(train_dfs, targetLabels, test_size=0.4)
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(train_values, train_target)
predictions = knn.predict(test_values)
print("KNN Accuracy = " + str(accuracy_score(test_target, predictions, normalize=True)))

# Test Gaussian Naive Bayes
train_values, test_values, train_target, test_target = train_test_split(train_dfs, targetLabels, test_size=0.4)
gnb = GaussianNB()
gnb.fit(train_values, train_target)
predictions = gnb.predict(test_values)
print("Naive Bayes Accuracy = " + str(accuracy_score(test_target, predictions, normalize=True)))


# Classify queries using decision trees
decisionTreeClassifier = DecisionTreeClassifier(criterion='entropy')
decisionTreeClassifier.fit(train_dfs, targetLabels)
predictions = decisionTreeClassifier.predict(queries_dfs)

# Classify queries using knn
# knn = KNeighborsClassifier(n_neighbors=11)
# knn.fit(train_dfs, targetLabels)
# predictions = knn.predict(queries_dfs)

# Classify queries using naive bayes
# gnb = GaussianNB()
# gnb.fit(train_dfs, targetLabels)
# predictions = gnb.predict(queries_dfs)


# Write predictions to file
predictions = '"' + predictions + '"'
predictions_df = pd.DataFrame(data=predictions, index=queries['id'].as_matrix())
predictions_df.to_csv('predictions.txt', sep=',', header=None, quoting=csv.QUOTE_NONE)
