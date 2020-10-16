import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reading Data from CSV and removing unwanted index Column
df = pd.read_csv('data/Train.csv')
df = df.drop(columns=['Unnamed: 0'],)

label_array = np.array(df['Label'])
(_, counts) = np.unique(label_array, return_counts=True)

plt.hist(counts,)
plt.show()
