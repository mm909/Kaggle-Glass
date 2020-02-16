import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

load = 1
glass = pd.read_csv("glass.csv")

# Split labels
train_, test_ = train_test_split(glass, test_size = 0.15, random_state = 42)
train_labels = train_["Type"]
train_ = train_.drop(columns = ["Type"])
test_labels = test_["Type"]
test_ = test_.drop(columns = ["Type"])
train = train_

if load:
    currPredict = joblib.load('dtc.pkl')
else:
    # # pg.175
    param_grid = [
        {'max_depth':[1,3,5,7,9,11,13,15,17], 'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']}
    ]

    tree_clf = DecisionTreeClassifier()
    grid_search = GridSearchCV(tree_clf, param_grid, cv = 5, return_train_score=True, n_jobs=-1, iid = True)
    grid_search.fit(train, train_labels)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    currPredict = grid_search.best_estimator_

count = 0
tree_predict = currPredict.predict(test_)
for i, prediction in enumerate(tree_predict):
    if prediction == test_labels.iloc[i]:
        count += 1
    pass
print(count/len(test_))


export_graphviz(
    currPredict,
    out_file = 'glass_tree.dot',
    feature_names = glass.columns[:9]
)

joblib.dump(currPredict, "dtc.pkl")
