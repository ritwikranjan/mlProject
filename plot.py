import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reading Data from CSV and removing unwanted index Column
df = pd.read_csv('data/Train.csv')
df = df.drop(columns=['Unnamed: 0'],)

label_array = np.array(df['Label'])
(emojis, counts) = np.unique(label_array, return_counts=True)

list_count = list(counts)
list_emoji = list(emojis)

plt.style.use('dark_background')
plt.bar(list_emoji, list_count, tick_label=list_emoji)
plt.xlabel(xlabel='Emoji Labels')
plt.ylabel(ylabel='Count')
plt.title('Frequency of Each Emoji Data')
plt.show()
