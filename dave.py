"""
-------------------------------------------------------
Dave Recommendation System Edition
TODO:
    - Decide whether we are making more than one recommendation
-------------------------------------------------------
Author:  Einstein Oyewole
ID:      180517070
Email:   oyew7070@mylaurier.ca
__updated__ = ""
-------------------------------------------------------
"""


# Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from utility import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD, MiniBatchSparsePCA
from scipy.spatial.distance import pdist, euclidean, squareform
import random
from numpy.linalg import norm

# Constants
THRESHOLD_FOR_LIKES = 2.5
EXPLORATION_RATE =  0.1
COLLABORATIVE_FILT_RATE = 0.3


def dimension_reduction(data):
    """
    -------------------------------------------------------
    Feature extraction using Latent Semantic analysis
    Makes sure we have enough components to capture 80% of the variance
    -------------------------------------------------------
    Parameters:
       data: pandas dataframe
    Returns:
       data in a reduced dimension (pandas DatFrame)
    -------------------------------------------------------
    """
    stop = False
    n_comp = int(0.25 * data.shape[1])
    while not stop:
        svd = TruncatedSVD(n_components=n_comp, n_iter=10)
        fit = svd.fit(data)
        stop = sum(fit.explained_variance_ratio_) >= 0.8  #
        if not stop:
            n_comp += 2
    new_data = svd.transform(data)
    return pd.DataFrame(data = new_data, index = data.index)


def similarity_func(u, v):
    return 1/(1+euclidean(u,v))


class Dave:
    """
    RL Recommender (Boost from gaming)
    """
    LEARNING_RATE = 0.8
    THRESHOLD_FOR_LIKES = 2.5
    EXPLORATION_RATE = 0.05
    COLLABORATIVE_FILT_RATE = 0.15

    def __init__(self, filename = None):
        """
        -------------------------------------------------------
        Initializes the class dave. Either loads up the pickle file
        or from start using utility `load_model()`
        -------------------------------------------------------
        Parameters:
           filename: pickle file
        Returns:
           None
        -------------------------------------------------------
        """

        if filename is None:
            self.movies, self.data = load_data()
            self.Q = pd.DataFrame(data=np.zeros((len(self.data.userId.unique()), len(self.movies.index.unique()))),
                                    columns=self.movies.index.unique(),
                                    index=self.data.userId.unique())
            self.movies = dimension_reduction(self.movies)
            n = pdist(self.movies, similarity_func)
            self.movies_sim = pd.DataFrame(squareform(n), index=self.movies.index, columns = self.movies.index)

            # self.build_Q()
            userd = list(self.data.userId.unique())
            val = [False] * len(userd)
            self.to_calc = {userd[i]: val[i] for i in range(len(userd))}
            self.data.drop_duplicates(inplace=True)
            # type casting
            self.movies: pd.DataFrame
            self.data: pd.DataFrame
            self.Q: pd.DataFrame
            self.to_calc: dict

    @staticmethod
    def get_actions(movies, state):
        """
        -------------------------------------------------------
        Returns a set of movies that has not been watched
            Movies not in the state
        -------------------------------------------------------
        Parameters:
           movies: set of all movies
           state: set of movies that has been watched/ rated
        Returns:
           actions: set of unwatched/unranked movies
        -------------------------------------------------------
        """
        return set(movies).difference(set(state))

    def get_state(self, userId) -> dict:
        """
        -------------------------------------------------------
        Gets movies a user has watched and returns a weighted rating with respect
        to the time the rating was provided
        -------------------------------------------------------
        Parameters:
           userId: userId also represent the current state
        Returns:
           movies: (dict -> {movieId: rating})
        -------------------------------------------------------
        """
        # Old way
        # tmp = self.data.loc[self.data["userId"] == userId, ["movieId", "rating", "timestamp_rating"]]
        # tmp = tmp.drop_duplicates()
        # # tmp["weighted_ratings"] = tmp.apply(
        # #     lambda x: weight_time(x['timestamp_rating'], x["rating"], tmp["timestamp_rating"].max()), axis=1)
        # state = tmp[['movieId', 'rating']].set_index('movieId').to_dict()['rating']

        # New way
        state = self.Q.loc[userId, self.Q.loc[userId].isna()].to_dict()
        return state

    def build_Q(self):
        """
        -------------------------------------------------------
        Builds the Q matrix : a Movie-User dataframe
        (1) Original Q matrix
        (2) Expected Q ratings matrix
        (3) Q matrix with some learning rate
        -------------------------------------------------------
        """
        # populating matrix with Q-value
        mov_cols = list(self.Q.columns)
        for userId, _ in tqdm(self.Q.iterrows()):
            state = self.get_state(userId)
            for mov in mov_cols:
                if mov in state.keys():
                    self.Q.loc[userId, mov] = np.NAN
                else:
                    self.Q.loc[userId, mov] = self.mov_simi(state, mov)  # (2), (1)
                    # self.Q.loc[userId, mov] = round((1-self.LEARNING_RATE)*self.Q.loc[userId, mov] + self.LEARNING_RATE*self.mov_simi(state, mov), 2)  # (3)

    def mov_simi(self, state: dict, movie: int):
        """
        -------------------------------------------------------
        We find the similarity of a movie and each of the previously rated movies
        (1) We generate a score by discounting the ratings deviation from THRESHOLD_FOR_LIKES
            i.e. negatives for less than 2.5 and positive for greater than 2.5
        (2) We generate the expected rating as in Multi-Armed Bandit
        -------------------------------------------------------
        Parameters:
           state: movies that has been rated and its weighted ratings (dict)
           movie: movie id of movie to compare with (int)
        Returns:
           res: (1) aggregated score (float)
                (2) Expected rating
                    (i) : similarity*rating
                    (ii): normal distribution of rating (later on)
        -------------------------------------------------------
        """
        res = 0
        m1 = np.array(self.movies.loc[movie])
        for mov, rating in state.items():
            m2 = np.array(self.movies.loc[mov])
            # similarity = cosine(m1,m2)
            # dist = pdist([m1,m2],metric="minkowski", p=5)[0]
            similarity = self.movies_sim.loc[mov, movie]
            # similarity = round(1 / (1 + dist),2)  # Euclidean similarity measure
            # similarity = 1- dist
            tmp = round(((rating - THRESHOLD_FOR_LIKES) / 7.5),2) * similarity  # discounts for how much further from 2.5 (1)
            # tmp = similarity*rating  # (2)
            res += tmp
        # res /= len(state)  # (2)
        return res

    def rewards(self, userId, rating, movie_rated):
        """
        -------------------------------------------------------
        We use the ratings as the rewards and propagate it in the users state
        -------------------------------------------------------
        Parameters:
           userId: user who provided the rating (int)
           rating: rating for corresponding movie (float 0.5 ≤ rating ≤ 5.0)
           movie_rated: movie the user rated (int)
        Returns:
           None
        -------------------------------------------------------
        """
        if not self.to_calc[userId]:
            # print("here")
            self.to_calc[userId]= True
        new_row = {'movieId': movie_rated, 'userId' : userId, 'rating': rating, 'timestamp_rating': np.datetime64('now')}
        self.data = self.data.append(new_row, ignore_index = True)
        self.propagate_reward(reward=rating, userId=userId, movie_rated=movie_rated)
        return

    def propagate_reward(self, reward, userId, movie_rated):
        """
        -------------------------------------------------------
        when something new is ranked
            (1) aggregated score (float)
            (2) Expected rating : need to lessen effect of low similarity
            (3) Q matrix with some learning rate
        -------------------------------------------------------
        Parameters:
           reward: the rating for mov_r  (float 0.5 ≤ rating ≤ 5.0)
           movie_rated: movie rated by the user; what we are learning from (int)
           userId: user (int)
        Returns:
           None
        -------------------------------------------------------
        """
        n = len(self.get_state(userId))+1
        mov_cols = list(self.Q.columns)
        m1 = np.array(self.movies.loc[movie_rated])
        # Update the Q table
        for mov in mov_cols:
            if mov == movie_rated:
                self.Q.loc[userId, mov] = np.nan  # mov_r should now be a part of state
            elif self.Q.loc[userId, mov] != np.nan:
                m2 = np.array(self.movies.loc[mov])
                # similarity = cosine(m1,m2)
                # dist = pdist([m1,m2],metric="minkowski", p=5)[0]
                similarity = self.movies_sim.loc[mov,movie_rated]
                # similarity = 1 / (1 + dist)  # Euclidean similarity measure
                # similarity = 1-dist
                tmp = ((reward - THRESHOLD_FOR_LIKES) / 7.5) * similarity  # discounts for how much further from 2.5 (1) (3)
                self.Q.loc[userId, mov] += tmp  # (1)
                # R = similarity*reward  # (2)
                # self.Q.loc[userId, mov] = ((n-1)*self.Q.loc[userId, mov] + R)/n  # (2)
                # self.Q.loc[userId, mov] = (1-self.LEARNING_RATE)*self.Q.loc[userId, mov] + self.LEARNING_RATE*tmp  # (3)
        # print(self.Q.loc[userId, [544, 318, 5312, 3363, 548, 2406, 6662, 904, 908, 1617, 1234, 3798, 1086]])

    def recommend(self, userId, count):
        """
        -------------------------------------------------------
        Chooses a movie/action for a user.
            Can choose a random movie
            Can choose the popular movies
            Can also choose specific movies to users
        -------------------------------------------------------
        Parameters:
           userId: user we are making a recommendation for
           count: maximum Number of movies to recommend
        Returns:
            action: movieId a movie to watch
        -------------------------------------------------------
        """
        state = self.get_state(userId)
        movies = list(self.movies.index)
        actions = self.get_actions(movies, state)
        l = random.uniform(0, 1)

        if l < self.EXPLORATION_RATE:
            """
            Explore: select a random choice
            """
            print("random movies [to change up things :) ]")
            max_movie = list(actions)
        elif self.EXPLORATION_RATE < l < self.COLLABORATIVE_FILT_RATE:
            """
            Explore: select a popular choice
            """
            print("popular well-rated movies")
            # do something
            tmp = self.data[self.data.movieId.isin(actions)]
            tmp_grp = tmp[["movieId", "rating"]]
            tmp_grp["Count"] = 1
            tmp_grp = tmp_grp.groupby(by=["movieId"]).agg({'rating': 'mean', 'Count': 'size'})
            tmp_grp = tmp_grp.reset_index()
            # max_val = tmp_grp.rating.max()
            res = tmp_grp.loc[tmp_grp.rating >= 4.0, ]
            res = res.loc[res.Count >= res.Count.quantile(q=0.9),]
            max_movie = list(res.movieId)
        else:
            """
            Choose best based on feedback
            """
            print("User recommendations")
            if not self.to_calc[userId]:  # if we have not calculated the values
                mov_cols = list(self.Q.columns)
                print(" - Thinking? - Loading - •••")
                for mov in tqdm(mov_cols):
                    self.Q.loc[userId, mov] = self.mov_simi(state, mov)
                self.to_calc[userId] = not self.to_calc[userId]
            max_val = self.Q.loc[userId].max()
            max_movie = [index for index, value in dict(self.Q.loc[userId]).items() if value == max_val]
        random.shuffle(max_movie)
        # Choosing movies
        action = max_movie[:count]
        return action

    def save_model(self, filename = "dave_rec_edition"):
        """
        -------------------------------------------------------
        saves/serializes the object into a file:
        WARNING: can be very large
        -------------------------------------------------------
        Parameters:
          filename: to save as (str)
        Returns:
           None
        -------------------------------------------------------
        """
        with open(filename, "wb") as fp:
            pickle.dump(self, fp)
        return

    @staticmethod
    def load_model(filename):
        """
        -------------------------------------------------------
        Loads instance from a file
        -------------------------------------------------------
        Parameters:
           filename: file to load from (str)
        Returns:
           None
        -------------------------------------------------------
        """
        with open(filename, "rb") as fp:
            class_obj = pickle.load(fp)
        return class_obj

    def add_user(self):
        """
        -------------------------------------------------------
        Adds a user to model
        -------------------------------------------------------
        Parameters:
        Returns:
           userId
        -------------------------------------------------------
        """
        new_userId = max(self.to_calc.keys()) + 1
        # self.to_calc[new_userId] = False

        # Build user's Q
        mov_cols = list(self.Q.columns)
        self.Q.loc[new_userId] = [0]*len(mov_cols)
        self.to_calc[new_userId] = True
        return new_userId

    def mAP(self, userId):
        """
        -------------------------------------------------------
        An Evaluation function:
            evaluates whether the recommended movie is relevant
            Uses precision.
        -------------------------------------------------------
        Parameters:
           userId: (int)
        Returns:
           score: a precision score of how relevant a movie is
        -------------------------------------------------------
        """
        # User's Q
        # pre_rating = self.Q.loc[userId]
        does_not_matter = self.rid_some_movies(userId)
        self.Q.loc[userId, does_not_matter] = np.nan
        # pre_rating = pre_rating.dropna()

        # User's data
        user_data = self.data.loc[self.data["userId"] == userId, ]
        train, test = train_test_split(user_data, test_size=0.4)

        #
        test["relevant"] = test["rating"] >= test["rating"].mean()
        test["relevant"] = test["relevant"].astype(int)
        rel = set(test.loc[test["relevant"] == 1, "movieId"].to_list())

        print(rel)
        # training
        for index, row in train.iterrows():

            self.rewards(userId=2, rating=row["rating"], movie_rated=row["movieId"])
            print(self.to_calc[userId], " ", row["rating"])
            print(self.Q.loc[userId,[544, 318, 5312, 3363, 548, 2406, 6662, 904, 908, 1617, 1234, 3798, 1086]])

        print(f"Done training, Rewards : {self.to_calc[userId]}")
        pred = set(self.recommend(userId = userId, count=len(rel)))

        precision = len(pred & rel)/len(pred)
        # reset Q
        self.Q.loc[userId] = [0] * len(self.movies.index)
        return precision

    def rid_some_movies(self, userId):
        """
        -------------------------------------------------------
        Helper function to limit the scope of movies rated.
        It returns every other movie that was not rated by user Id
        -------------------------------------------------------
        Parameters:
           userID
        Returns:
           actions: list of movies not to be considered
        -------------------------------------------------------
        """
        movies = list(self.movies.index)
        state = self.data.loc[self.data["userId"] == userId, "movieId"].unique()
        actions = self.get_actions(movies, state)
        return actions