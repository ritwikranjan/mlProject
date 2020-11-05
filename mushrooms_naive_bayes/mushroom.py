import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('mushrooms.csv')
le = LabelEncoder()

df = df.apply(le.fit_transform)

Y = df.values[:, 0]
X = df.values[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


def prior_prob(y_train, cat):
    num = np.sum(y_train == cat)
    den = y_train.shape[0]
    return num/float(den)


def cond_prob(x_train, y_train, feature_col, feature_val, cat):

    x_train = x_train[y_train == cat]

    num = np.sum(x_train[:, feature_col] == feature_val)
    den = np.sum(y_train == cat)

    return num/float(den)


def predict(x_train, y_train, test_sample):

    post_prob = list()
    features = x_train.shape[1]
    classes = np.unique(y_train)

    for label in classes:

        likelihood = 1.0

        for f in range(features):
            cond_probability = cond_prob(x_train, y_train, f, test_sample[f], label)
            likelihood *= cond_probability

        post_probability = likelihood*prior_prob(y_train, label)
        post_prob.append(post_probability)

    return np.argmax(post_prob)


predictions = list()
for test_data in X_test:
    predictions.append(predict(X_train, Y_train, test_data))

matrix = metrics.confusion_matrix(predictions, Y_test)
