"""
-------------------------------------------------------
Utility functions
-------------------------------------------------------
Author:  Einstein Oyewole
ID:      180517070
Email:   oyew7070@mylaurier.ca
__updated__ = ""
-------------------------------------------------------
"""


# Imports
import pandas as pd

from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# Constants
nltk.download('stopwords')
STOPWORDS = stopwords.words('english')


def weight_time(timestamp, ratings, max_date):
    """
    Exponential decay of the ratings based on the time difference,
    Adjusts ratings for current ratings
    Fixed Parameter:
    2: constant scaling
    5: time constant decay"""
    days = (max_date - timestamp).days
    weights = 2*np.exp(-days/5)
    return round(ratings*weights,2)


def load_data():
    """
    -------------------------------------------------------
    [Function Description]
    -------------------------------------------------------
    Parameters:
       [parameter name - parameter description (parameter type and constraints)]
    Returns:
       [return value name - return value description (return value type)]
    -------------------------------------------------------
    """
    # Tags data : user dependent
    tags = pd.read_csv("ml-latest-small/tags.csv")

    # Movie data set
    movies = pd.read_csv("ml-latest-small/movies.csv")
    s = movies.genres.str.split("|")
    # pd.get_dummies(movies.genres.explode()).sum(level=0)
    tmp = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    tmp = tmp.astype("category")
    movies = pd.concat([movies, tmp], axis=1)
    movies = movies.drop("genres", axis=1)
    del s
    del tmp

    # ratings data set
    ratings = pd.read_csv("ml-latest-small/ratings.csv")

    # Joining the dataset
    tag_ratings = pd.merge(ratings, tags, on=["userId", "movieId"], how="left")
    tag_ratings["tag"] = tag_ratings["tag"].fillna("")
    tag_ratings = tag_ratings.drop("timestamp_y", axis=1)
    tag_ratings["timestamp_x"] = tag_ratings["timestamp_x"].apply(datetime.fromtimestamp)
    #tag_ratings["timestamp_y"] = tag_ratings["timestamp_y"].apply(datetime.fromtimestamp)
    tag_ratings = tag_ratings.rename(columns={"timestamp_x": "timestamp_rating"})
    data = pd.merge(tag_ratings, movies, on="movieId")
    del tag_ratings

    # aggregating tags for a movie and extracting its features (using bag of words)
    movie_tags = data[["movieId", "tag"]]
    movie_tags["tag"] = movie_tags.groupby(by=["movieId"])["tag"].transform(lambda x: " ".join(x))
    movie_tags = movie_tags.drop_duplicates()
    cv = CountVectorizer(max_features=50, analyzer="word", stop_words=STOPWORDS)
    tmp = cv.fit_transform(movie_tags["tag"])
    df_tp = pd.DataFrame(data=tmp.toarray(), columns=cv.get_feature_names(), index=movie_tags['movieId'])
    movie_tags = df_tp.reset_index()
    del df_tp
    del tmp
    del cv

    # Join movie tags with movies data
    movies = pd.merge(movies, movie_tags, on="movieId", how="left")
    movies.set_index('movieId', inplace=True)
    movies.drop("title", inplace=True, axis = 1)
    movies = movies.fillna(0)
    del movie_tags
    return movies, data[['movieId', 'userId', 'rating', 'timestamp_rating']]




