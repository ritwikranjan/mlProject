import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd

# Reading Data from CSV and removing unwanted index Column
df = pd.read_csv('data/Train.csv')
df = df.drop(columns=['Unnamed: 0'], )

corpus = []

for i in range(70000):
    text = re.sub('[^a-zA-Z]', ' ', df['TEXT'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    review = ' '.join(text)
    corpus.append(review)

print(corpus)
