import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

tmdb_api_url = 'https://api.themoviedb.org/3'
tmdb_posters_url = 'https://image.tmdb.org/t/p/w185'
tmdb_token = '10cab06b92e0570e93687deba6fbffc0'
tmdb_attribution_logo = 'https://www.themoviedb.org/assets/2/v4/logos/v2/' \
                        'blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg'

top_rated_movies_df = pd.read_json(tmdb_api_url + '/discover/movie' + '?api_key=' + tmdb_token +
                                   '&certification_country=US&certification.lte=PG-13&vote_count.gte=1000' +
                                   '&sort_by=vote_average.desc')
movies_to_rate_df = pd.json_normalize(top_rated_movies_df['results'])
movies_to_rate_df = movies_to_rate_df[movies_to_rate_df['original_language'] == 'en']

st.header('Tag-Based Movie Recommender')
""" 
  This recommender finds movies similar to movies you\'ve rated positively.
  
  Use the slider to rate the movie on a scale of 1 to 5, 5 being the highest, then click 'Rate'
  to submit your rating.

  After rating a few movies (or all of them!), click 'Show recommended movies'
  to see the movies most similar to movies you like.
"""

mov1_title, mov2_title, mov3_title, mov4_title = st.columns(4)
with mov1_title:
    mov1_title_container = st.container()
with mov2_title:
    mov2_title_container = st.container()
with mov3_title:
    mov3_title_container = st.container()
with mov4_title:
    mov4_title_container = st.container()

mov1, mov2, mov3, mov4 = st.columns(4)
with mov1:
    mov1_container = st.container()
with mov2:
    mov2_container = st.container()
with mov3:
    mov3_container = st.container()
with mov4:
    mov4_container = st.container()


# Function to cache CSV dataframes for Streamlit
@st.experimental_memo
def cache_csv(filename):
    _cached_csv = pd.read_csv(filename)
    return _cached_csv


# Read in the movies, tags, and links dataframes
movies_df = cache_csv('ml-25m/movies.csv')
rgs_df = cache_csv('ml-25m/reduced-genome-scores.csv')
links_df = cache_csv('ml-25m/links.csv')
movie_tags_df = cache_csv('ml-25m/tags.csv')
id_to_tag_df = cache_csv('ml-25m/genome-tags.csv')

# Generate replacement dictionaries for swapping the TMDB movie ID with the MovieLens movie ID, movieId with title,
# and tagId with tag
id_repl = links_df.set_index('tmdbId')['movieId'].to_dict()
movie_id_to_title = movies_df.set_index('movieId')['title'].to_dict()
tag_id_to_tag = id_to_tag_df.set_index('tagId')['tag'].to_dict()

# Initialize the session state variables.
st_session_state = {'interactive_user_ratings': [], 'mov1_i': 0, 'mov2_i': 1, 'mov3_i': 2, 'mov4_i': 3,
                    'mov1_slider': 3, 'mov2_slider': 3, 'mov3_slider': 3, 'mov4_slider': 3,
                    'mov1_done_rating': False, 'mov2_done_rating': False, 'mov3_done_rating': False,
                    'mov4_done_rating': False}
for k, v in st_session_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Remove 'original' tag, ID 742
rgs_df.drop('742', axis=1, inplace=True)


# Function to rate the movie. To account for race condition when the user sets the slider and clicks 'Rate'
# before the app has updated, if the rating is not set, display an error.
# After submitting the rating, we set the slider back to 3 using the session state variable.
def rate_movie(position, movie_id, rating):
    if rating is None:
        st.error('No rating provided.')
    else:
        st.session_state.interactive_user_ratings.append({'movieId': movie_id, 'rating': rating})
        if position == 'mov1':
            if (st.session_state.mov1_i + 4) < movies_to_rate_df.shape[0]:
                st.session_state.mov1_i += 4
                st.session_state.mov1_slider = 3
            else:
                st.session_state.mov1_done_rating = True
        elif position == 'mov2':
            if (st.session_state.mov2_i + 4) < movies_to_rate_df.shape[0]:
                st.session_state.mov2_i += 4
                st.session_state.mov2_slider = 3
            else:
                st.session_state.mov2_done_rating = True
        elif position == 'mov3':
            if (st.session_state.mov3_i + 4) < movies_to_rate_df.shape[0]:
                st.session_state.mov3_i += 4
                st.session_state.mov3_slider = 3
            else:
                st.session_state.mov3_done_rating = True
        elif position == 'mov4':
            if (st.session_state.mov4_i + 4) < movies_to_rate_df.shape[0]:
                st.session_state.mov4_i += 4
                st.session_state.mov4_slider = 3
            else:
                st.session_state.mov4_done_rating = True


# Set up the columns of movies to rate. If there are no movies to rate, clear the container. Else get the next
# movie to rate. Fetch poster images from TMDB API.
with mov1:
    if st.session_state.mov1_done_rating:
        mov1_container.empty()
        for _ in range(10):
            mov1_container.text('')
        mov1_container.markdown('_Done rating_')
    else:
        mov1_title_container.markdown('_' + movies_to_rate_df.iloc[st.session_state.mov1_i]['title'] + '_')
        mov1_container.image(tmdb_posters_url + movies_to_rate_df.iloc[st.session_state.mov1_i]['poster_path'])
        mov1_rating = mov1_container.slider('Rating', 1, 5, key='mov1_slider')
        mov1_rate = mov1_container.button('Rate', key='mov1_button', on_click=rate_movie,
                                          args=('mov1', movies_to_rate_df.iloc[st.session_state.mov1_i]['id'],
                                                mov1_rating - 3))
with mov2:
    if st.session_state.mov2_done_rating:
        mov2_container.empty()
        for _ in range(10):
            mov2_container.text('')
        mov2_container.markdown('_Done rating_')
    else:
        mov2_title_container.markdown('_' + movies_to_rate_df.iloc[st.session_state.mov2_i]['title'] + '_')
        mov2_container.image(tmdb_posters_url + movies_to_rate_df.iloc[st.session_state.mov2_i]['poster_path'])
        mov2_rating = mov2_container.slider('Rating', 1, 5, key='mov2_slider')
        mov2_rate = mov2_container.button('Rate', key='mov2_button', on_click=rate_movie,
                                          args=('mov2', movies_to_rate_df.iloc[st.session_state.mov2_i]['id'],
                                                mov2_rating - 3))
with mov3:
    if st.session_state.mov3_done_rating:
        mov3_container.empty()
        for _ in range(10):
            mov3_container.text('')
        mov3_container.markdown('_Done rating_')
    else:
        mov3_title_container.markdown('_' + movies_to_rate_df.iloc[st.session_state.mov3_i]['title'] + '_')
        mov3_container.image(tmdb_posters_url + movies_to_rate_df.iloc[st.session_state.mov3_i]['poster_path'])
        mov3_rating = mov3_container.slider('Rating', 1, 5, key='mov3_slider')
        mov3_rate = mov3_container.button('Rate', key='mov3_button', on_click=rate_movie,
                                          args=('mov3', movies_to_rate_df.iloc[st.session_state.mov3_i]['id'],
                                                mov3_rating - 3))
with mov4:
    if st.session_state.mov4_done_rating:
        mov4_container.empty()
        for _ in range(10):
            mov4_container.text('')
        mov4_container.markdown('_Done rating_')
    else:
        mov4_title_container.markdown('_' + movies_to_rate_df.iloc[st.session_state.mov4_i]['title'] + '_')
        mov4_container.image(tmdb_posters_url + movies_to_rate_df.iloc[st.session_state.mov4_i]['poster_path'])
        mov4_rating = mov4_container.slider('Rating', 1, 5, key='mov4_slider')
        mov4_rate = mov4_container.button('Rate', key='mov4_button', on_click=rate_movie,
                                          args=('mov4', movies_to_rate_df.iloc[st.session_state.mov4_i]['id'],
                                                mov4_rating - 3))


# Function to generate the user tag profile and compare it against the movie database. First, we multiply the tag
# value (1 or 0) for each movie the user rated by the rating the user gave, which was converted earlier to a scale of
# -2 to 2 by subtracting 3 from the rating. This weights the tag by the rating the user gave.
#
# Then we add up the weighted tag value for each tag. If the result is a net positive, we set the tag value to 1, if
# it's 0 or less, we set the tag value to 0. This allows tags the user probably doesn't like to be removed.
#
# For example, if a tag is associated with one movie the user rated positively, but several movies the user rated
# negatively, the tag should be removed from the profile using this method. We use cosine_similarity to generate the
# similarity score for each movie as compared to the user tag profile, and then print out the top 30 movies ordered by
# similarity score.
def recommend_movies():
    if not st.session_state.interactive_user_ratings:
        st.error("Please rate some movies first")
        return
    user_profile_df = pd.DataFrame(st.session_state.interactive_user_ratings)
    user_profile_df.drop_duplicates(inplace=True, subset='movieId', keep='last')
    user_profile_df['movieId'].replace(id_repl, inplace=True)

    user_tag_profile = rgs_df[rgs_df['movieId'].isin(user_profile_df['movieId'].tolist())].reset_index(drop=True)
    user_tag_profile.drop('movieId', axis=1, inplace=True)
    user_tag_profile = user_tag_profile.mul(user_profile_df['rating'], axis=0)

    user_profile = pd.DataFrame(user_tag_profile.sum(axis=0).map(lambda x: 1 if x > 0 else 0))
    user_profile = user_profile.transpose()

    rgs_df.set_index('movieId', drop=True, inplace=True)
    rgs_df['cos_sim'] = cosine_similarity(rgs_df, user_profile).reshape(-1)
    rgs_df.sort_values(by='cos_sim', ascending=False, inplace=True)

    recommendations_df = movies_df[movies_df['movieId'].isin(rgs_df.head(30).index)].reset_index(drop=True)
    recommendations_df = recommendations_df[~recommendations_df['movieId'].isin(user_profile_df['movieId'])]

    with recommendation_container:
        st.table(recommendations_df['title'])


show_recs = st.button('Show recommended movies', on_click=recommend_movies)
recommendation_container = st.container()
recommendation_container.write('')

footer = st.container()
footer.markdown('This app uses MovieLens data from the GroupLens Research Group at the University of Minnesota. '
                'This application is not affiliated with or endorsed by the University of Minnesota.')
footer.markdown('[MovieLens](https://grouplens.org/datasets/movielens/)')
footer.markdown('This product uses the TMDB API but is not endorsed or certified by TMDB.')
footer.image(tmdb_attribution_logo, width=128)
footer.markdown('[TMDB](https://www.themoviedb.org/)')
