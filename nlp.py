import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd

# Reading Data from CSV and removing unwanted index Column
df = pd.read_csv('data/Train.csv')
df = df.drop(columns=['Unnamed: 0'], )
corpus = list()
ps = PorterStemmer()
sw = set(stopwords.words('english'))
nsw = {'no', 'not', 'what', 'when', 'where', 'how', 'whom', 'which'}

for i in range(70000):
    text = df['TEXT'][i]
    text = text.lower()
    text = re.findall('[a-z]+', text)
    text = [ps.stem(word) for word in text if word not in sw or word in nsw]
    text = ' '.join(text)
    corpus.append(text)

refined_df = pd.DataFrame(corpus)
refined_df = refined_df.join(df['Label'])
refined_df.columns = ['text', 'emoji_label']

refined_df.to_csv('data/cleaned_train.csv', index=False)

