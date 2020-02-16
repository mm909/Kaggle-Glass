# Implementing algorithms from Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow with custom dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    # print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# Ri Na Mg Al Si K Ca Ba Fe Type
glass = pd.read_csv("glass.csv")

# # pg.47 - 48
# print(glass.head())
# print(glass.info())
# print(glass.describe())

# # pg.49
# glass.hist(bins=50)
# plt.show()

# Split labels
train_, test_ = train_test_split(glass, test_size = 0.2, random_state = 42)
train_labels = train_["Type"]
train_ = train_.drop(columns = ["Type"])
test_labels = test_["Type"]
test_ = test_.drop(columns = ["Type"])
# print(train_.head())
# print(train_labels.head())

# # pg.58
# corr_matrix = train_.corr()
# print(corr_matrix["Type"].sort_values(ascending = False))

# # pg.60
# attributes = ["RI", "Na", "Mg", "Al"]
# pd.plotting.scatter_matrix(glass_[attributes])
# plt.show()

# # pg.70
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

train = num_pipeline.fit_transform(train_)
test_ = num_pipeline.transform(test_)

# # pg.175
tree_clf = DecisionTreeClassifier(max_depth = 3)
tree_clf.fit(train, train_labels)

# # pg.75
forset_clf = RandomForestClassifier(n_estimators = 100, max_depth = 3)
forset_clf.fit(train, train_labels)

# # pg.192
log_clf = LogisticRegression(multi_class = 'auto', solver = 'lbfgs')
rnd_clf = RandomForestClassifier(n_estimators = 100, max_depth = 5)
svm_clf = SVC(gamma = 5, C = 1000, probability = True)

voting_clf = VotingClassifier(
    estimators = [('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting = 'soft'
)
voting_clf.fit(train, train_labels)

# # # pg.73
# scores = cross_val_score(tree_clf, train, train_labels, scoring = "neg_mean_squared_error", cv = 5)
# forest_scores = cross_val_score(forset_clf, train, train_labels, scoring = "neg_mean_squared_error", cv = 5)
# voting_scores = cross_val_score(voting_clf, train, train_labels, scoring = "neg_mean_squared_error", cv = 5)
#
# tree_rmse_scores = np.sqrt(-scores)
# display_scores(tree_rmse_scores)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)
# voting_rmse_scores = np.sqrt(-voting_scores)
# display_scores(voting_rmse_scores)
#
# count = 0
# tree_predict = tree_clf.predict(train)
# for i, prediction in enumerate(tree_predict):
#     if prediction == train_labels.iloc[i]:
#         count += 1
#     pass
# print(count/len(train))
#
# count = 0
# forest_predict = forset_clf.predict(train)
# for i, prediction in enumerate(forest_predict):
#     if prediction == train_labels.iloc[i]:
#         count += 1
#     pass
# print(count/len(train))
#
# count = 0
# voting_predict = voting_clf.predict(train)
# for i, prediction in enumerate(voting_predict):
#     if prediction == train_labels.iloc[i]:
#         count += 1
#     pass
# print(count/len(train))

export_graphviz(
    tree_clf,
    out_file = 'glass_tree.dot',
    feature_names = glass.columns[:9]
)

count = 0
tree_predict = forset_clf.predict(test_)
for i, prediction in enumerate(tree_predict):
    if prediction == test_labels.iloc[i]:
        count += 1
    pass
print(count/len(test_))
