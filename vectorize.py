import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data/cleaned_data.csv')
df = df.dropna()
corpus = df['text']

#   cv = CountVectorizer(dtype=np.int8)


X_train, X_test, Y_train, Y_test = train_test_split(corpus, df['emoji_label'], test_size=0.1)

#   X_train = cv.fit_transform(X_train)
#   X_test = cv.transform(X_test)

'''
MNB = MultinomialNB()
MNB.fit(X_train, Y_train)
predict = MNB.predict(X_test)
accuracy = metrics.confusion_matrix(predict, Y_test)
sns.heatmap(accuracy)
plt.show()
print(metrics.accuracy_score(predict, Y_test))
print(metrics.f1_score(predict, Y_test, average='macro', labels=range(20)))
'''