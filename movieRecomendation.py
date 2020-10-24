import pandas as pd

# Data Generation
user_df = pd.read_csv('data/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'time'])
user_df = user_df.drop('time', axis=1)
movie_df = pd.read_csv('data/u.item', encoding='windows-1252', sep='|', header=None)
movie_df = movie_df.get([0, 1])
movie_df.columns = ['item_id', 'title']
df = pd.merge(user_df, movie_df, on='item_id')

# Data Analysis
analytics_df = pd.DataFrame(df.groupby('title')['rating'].mean())
analytics_df['count'] = df.groupby('title')['rating'].count()

# Movie Matrix: rating: user_id vs Movie
movie_matrix = df.pivot_table(values='rating', index='user_id', columns='title')


# Prediction Function
def predict_movies(movie_name):
    movie_corr = pd.DataFrame(movie_matrix.corrwith(movie_matrix[movie_name]).dropna(), columns=['Correlation'])
    movie_corr = movie_corr.join(analytics_df['count'])

    prediction = movie_corr[movie_corr['count'] > 100].sort_values(by='Correlation', ascending=False)
    return prediction


star_wars = predict_movies('Star Wars (1977)')
print(star_wars.head())
