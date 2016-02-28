"""
Fine-tune logistic regression parameters via terminal
since Jupyter notebook had problems finishing the jobs,
and the view went stale
"""

import os
import sys
import csv
import datetime
import itertools
import numpy as np
import pandas as pd
import random
import re
import sklearn
import time
from collections import defaultdict
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


kfood_data_file = 'data/kfood.txt'
kfood_data_file_raw = 'data/kfood.raw.txt'

catpat = re.compile(r'/(\w+)\.\d+\.jpg$')


def encode_category(df):
    # Encode categories as numerical index
    categories = sorted(set(df.category))
    cat2idx = {c: i for i, c in enumerate(categories)}
    df['category'] = df.category.apply(lambda c: cat2idx[c])
    return categories


def load_data():
    if os.path.exists(kfood_data_file):
        df = pd.read_csv(kfood_data_file)
        return df, df.columns[1:-1], encode_category(df)

    if not os.path.exists(kfood_data_file_raw):
        raise ValueError("data files don't exist")

    df = pd.read_csv(kfood_data_file_raw, header=None)
    feature_cols = list(df.columns[1:])
    for i in xrange(len(feature_cols)):
        feature_cols[i] = "feature{}".format(i)
    df.columns = ['filepath'] + feature_cols
    df['category'] = df.filepath.apply(lambda _fp: catpat.search(_fp).group(1))

    df.to_csv(kfood_data_file, index=False)
    return df, feature_cols, encode_category(df)

df, feature_cols, y_cats = load_data()

target_col = 'category'

X_all = df[feature_cols]  # feature values for all students
y_all = df[target_col]

# Split up the train and test data
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=test_size, random_state=0, stratify=y_all)

# Split up the filepaths for comparison (maybe useful later)
X_train_filepath, X_test_filepath = train_test_split(
    df['filepath'], test_size=test_size, random_state=0, stratify=y_all)

# Test to see that the train_test_split splits the filepaths
# the same way as the feature data
np.array_equal(pd.merge(pd.DataFrame(X_train_filepath, columns=['filepath']),
                        df,
                        how='left',
                        on='filepath')[feature_cols].values, X_train)




# # Logistic Regression ovr

# clf_lr = LogisticRegression(
#     penalty='l2', random_state=0, multi_class='ovr', n_jobs=4)

# best_score = None
# best_params = None
# for i in xrange(10):
#     if i < 3 or i % 10 == 0:
#         print "iteration {}".format(i)
#         start = time.time()
#     gs_clf = GSCV(
#         clf_lr,
#         {'tol': [0.0001, 0.001, 0.00001], 'C': [10.0, 1.0, .1, .01]},
#         scoring='log_loss',
#         cv=5,
#         n_jobs=1)
#     gs_clf.fit(X_train, y_train)
#     _score = accuracy_score(y_test, gs_clf.predict(X_test))
#     if best_score is None or best_score < _score:
#         best_score = _score
#         best_params = {
#             'C': gs_clf.best_estimator_.C,
#             'tol': gs_clf.best_estimator_.tol}
#     if i < 3:
#         end = time.time()
#         print "Each iteration time(secs): {:.3f}".format(end - start)

# print "OVR:"
# print best_score, best_params


# Logistic Regression multinomial
clf_lr2 = LogisticRegression(
    penalty='l2', random_state=0,
    multi_class='multinomial',
    solver='lbfgs', n_jobs=1)

best_score = None
best_params = None
for i in xrange(10):
    if i < 3 or i % 10 == 0:
        print "iteration {}".format(i)
        start = time.time()
    gs_clf = GSCV(
        clf_lr2,
        {'tol': [0.0001, 0.001, 0.00001],
         'C': [10.0, 1.0, .1, .01],
         'max_iter': [100, 50, 150]},
        scoring='log_loss',
        cv=5,
        n_jobs=1)
    gs_clf.fit(X_train, y_train)
    _score = accuracy_score(y_test, gs_clf.predict(X_test))
    if best_score is None or best_score < _score:
        best_score = _score
        best_params = {
            'C': gs_clf.best_estimator_.C,
            'max_iter': gs_clf.best_estimator_.max_iter,
            'tol': gs_clf.best_estimator_.tol}
    if i < 3:
        end = time.time()
        print "Each iteration time(secs): {:.3f}".format(end - start)

print "Multinomial:"
print best_score, best_params
