import pandas as pd


def create_sequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


def create_sequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


def rating_data_sequence_creation(user_path, movie_path, rating_path, sequence_length, step_size=1):
    """
    :param user_path: path of the user data
    :param movie_path: path of the movie data
    :param rating_path: path of the rating data
    :param sequence_length: the length of rating sequence
    :param step_size: the number of steps to move the sequence window
    :return: ratings sequence data with user demographics and sparse feature cardinality
    """
    users = pd.read_csv(user_path, sep='::', names=["user_id", "sex", "age_group", "occupation", "zip_code"])
    movies = pd.read_csv(movie_path, sep='::', names=["movie_id", "title", "genres"], encoding="ISO-8859-1")
    ratings = pd.read_csv(rating_path, sep='::', names=["user_id", "movie_id", "rating", "unix_timestamp"])
    # get the sparse cardinality
    sparse_cardinality = {
        'movie_id': max(movies["movie_id"]) + 1,
        'user_id': max(users["user_id"]) + 1,
        'sex': users['sex'].nunique(),
        'age_group': users['age_group'].nunique(),
        'occupation': users['occupation'].nunique(),
        'rating': 5
    }

    # reformat the data types
    ratings['unix_timestamp'] = pd.to_datetime(ratings['unix_timestamp'], unit='s')
    ratings["movie_id"] = ratings["movie_id"].astype(str)
    ratings["user_id"] = ratings["user_id"].astype(str)
    users["user_id"] = users["user_id"].astype(str)

    # create movie ratings data into sequence
    ratings_group = ratings.sort_values(by=['unix_timestamp']).groupby('user_id')
    ratings_data = pd.DataFrame(
        data={
            "user_id": list(ratings_group.groups.keys()),
            "movie_ids": list(ratings_group.movie_id.apply(list)),
            "ratings": list(ratings_group.rating.apply(list)),
            "timestamps": list(ratings_group.unix_timestamp.apply(list))
        }
    )
    ratings_data.movie_ids = ratings_data.movie_ids.apply(
        lambda ids: create_sequences(ids, sequence_length, step_size)
    )
    ratings_data.ratings = ratings_data.ratings.apply(
        lambda ids: create_sequences(ids, sequence_length, step_size)
    )
    del ratings_data["timestamps"]

    # join the user and movie data
    ratings_data_movie_sequence = ratings_data[["user_id", "movie_ids"]].explode(
        "movie_ids", ignore_index=True
    )
    rating_data_rating_sequence = ratings_data[["ratings"]].explode("ratings", ignore_index=True)
    ratings_sequence_data = pd.concat([ratings_data_movie_sequence, rating_data_rating_sequence], axis=1)
    ratings_sequence_data = ratings_sequence_data.merge(users, on="user_id", how="left")

    ratings_sequence_data.movie_ids = ratings_sequence_data.movie_ids.apply(
        lambda x: ",".join(x)
    )
    ratings_sequence_data.ratings = ratings_sequence_data.ratings.apply(
        lambda x: ",".join([str(v) for v in x])
    )
    del ratings_sequence_data["zip_code"]

    ratings_sequence_data.rename(
        columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
        inplace=True,
    )

    return ratings_sequence_data, sparse_cardinality



