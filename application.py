import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import yaml
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components
import pickle
import random

def SpotifyToken():
    with open("./SpotifyToken/SpotifyToken.yaml") as f:
        spotify_config = yaml.safe_load(f)

    # Authenticate Spotify client using credentials
    auth = SpotifyClientCredentials(client_id = spotify_config['client_id'], client_secret = spotify_config['client_secret'])
    sp = spotipy.Spotify(auth_manager = auth)
    return sp

def spotifyModel(url, model):
    sp = SpotifyToken()
    result = []

    uri = url.rsplit('/', 1)[-1].split('?')[0]
    if model == 'Spotify Model':
        sm = sp.recommendations(seed_tracks = [uri], limit = 25)
        result = [track['id'] for track in sm['tracks'][:25]]
        return result

def songsModel(url, model, max_gen = 3, same_art = 5):

    sp = SpotifyToken()

    uri = url.rsplit('/', 1)[-1].split('?')[0]

    # gets audio features for the track
    audio_ = pd.DataFrame(sp.audio_features([uri]))
    # Create a dictionary of old and new column names
    new_column_names = {'danceability': 'Danceability', 'energy': 'Energy', 'key': 'Key',
                        'loudness': 'Loudness', 'mode': 'Mode', 'speechiness': 'Speechiness',
                        'acousticness': 'Acousticness', 'instrumentalness': 'Instrumentalness',
                        'liveness': 'Liveness', 'valence': 'Valence', 'tempo': 'Tempo',
                        'type': 'Type', 'id': 'Id', 'uri': 'Uri', 'track_href': 'Track_Href',
                        'analysis_url': 'Analysis_Url', 'duration_ms': 'Duration(ms)',
                        'time_signature': 'Time_Signature', 'track_release_date': 'Track_Release_Date'}

    audio_.rename(columns = new_column_names, inplace = True)
    audio_.drop(columns = ['Type', 'Uri', 'Track_Href', 'Analysis_Url'], axis = 1, inplace = True)

    # gets the info about the track (track's name, artist(s), album, and popularity, ...)
    track_features = sp.tracks([uri])

    # track_ will have all details about the track
    track_features = sp.tracks([uri])

    track_ = pd.DataFrame()

    track_pop = pd.DataFrame(columns=['Track_Uri', 'Track_Release_Date', 'Track_Pop', 'Artist_Uri', 'Album_Uri'])
    track_pop.loc[0] = [uri, track_features['tracks'][0]['album']['release_date'], track_features['tracks'][0]["popularity"], track_features['tracks'][0]['artists'][0]['id'], track_features['tracks'][0]['album']['id']]
    track_ = pd.concat([track_, track_pop], axis = 0)

    # artist_ will have Artist_Uri, Artist_pop, and Artist_Genres
    artist_id = track_['Artist_Uri'][0]
    artist_df = pd.DataFrame([artist_id], columns = ['Artist_Uri'])
    artist_features = sp.artist(artist_id)
    artist_df['Artist_Pop'] = artist_features['popularity']
    artist_genres = artist_features['genres']
    artist_df["Artist_Genres"] = " ".join([re.sub(' ', '_', i) for i in artist_genres]) if artist_genres else "unknown"
    artist_ = artist_df

    test = pd.DataFrame(track_, columns = ['Track_Uri', 'Artist_Uri', 'Album_Uri'])
    test.rename(columns = {'Track_Uri': 'track_uri','Artist_Uri': 'artist_uri', 'Album_Uri': 'album_uri'}, inplace = True)

    test = pd.merge(test, audio_, left_on = "track_uri", right_on = "Id", how = 'outer')
    test = pd.merge(test, track_, left_on = "track_uri", right_on = "Track_Uri", how = 'outer')
    test = pd.merge(test, artist_, left_on = "artist_uri",right_on = "Artist_Uri", how = 'outer')

    test.drop(columns = ['Track_Uri', 'Artist_Uri_x','Artist_Uri_y', 'Album_Uri', 'Id'], axis = 1, inplace = True)
    test.rename(columns = {'track_uri': 'Track_Uri','artist_uri': 'Artist_Uri', 'album_uri': 'Album_Uri'}, inplace = True)

    test[['Track_Pop','Artist_Pop']] = test[['Track_Pop','Artist_Pop']].apply(lambda x: x/5).apply(np.floor)
    test['Track_Release_Date'] = test['Track_Release_Date'].apply(lambda x: int(x.split('-')[0])//50)

    test[['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness',
        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Time_Signature']] = test[['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness',
        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Time_Signature']].astype('float16')
    test[['Track_Release_Date', 'Track_Pop', 'Artist_Pop']] = test[['Track_Release_Date', 'Track_Pop', 'Artist_Pop']].astype('int8')

    test['Artist_Genres'] = test['Artist_Genres'].apply(lambda x: x.split(" "))

    tfidf = TfidfVectorizer(max_features = max_gen)  
    tfidf_matrix = tfidf.fit_transform(test['Artist_Genres'].apply(lambda x: " ".join(x)))

    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" +i for i in tfidf.get_feature_names_out()]
    genre_df = genre_df.astype('float16')

    test.drop(columns = ['Artist_Genres'], inplace = True)
    test = pd.concat([test.reset_index(drop = True), genre_df.reset_index(drop = True)], axis = 1)
    
    df = pd.read_csv('Data/Test.csv', header = 0)
    df['Artist_Genres'] = df['Artist_Genres'].apply(lambda x: x.split(" "))
    tfidf_matrix = tfidf.transform(df['Artist_Genres'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['Genre' + "|" + i for i in tfidf.get_feature_names_out()]

    genre_df=genre_df.astype('float16')
    df.drop(columns = ['Artist_Genres'], axis = 1, inplace = True)

    df = pd.concat([df.reset_index(drop = True), genre_df.reset_index(drop = True)], axis = 1)

    sc = pickle.load(open('Data/sc.sav','rb'))
    df.iloc[:, 3:19] = sc.transform(df.iloc[:, 3:19])
    test.iloc[:, 3:19] = sc.transform(test.iloc[:, 3:19])
    playvec = pd.DataFrame(test)

    if model == 'My Model':
        df['Sim1'] = cosine_similarity(df.drop(['Track_Uri', 'Artist_Uri', 'Album_Uri'], axis = 1), playvec.drop(['Track_Uri', 'Artist_Uri', 'Album_Uri'], axis = 1))
        #df['Sim2'] = cosine_similarity(df.iloc[:,16:-1], playvec.iloc[:,16:])
        #df['Sim3'] = cosine_similarity(df.iloc[:,19:-2], playvec.iloc[:,19:])
        df = df.sort_values(['Sim1'],ascending = False,kind='stable').groupby('Artist_Uri').head(same_art).head(25)
        top_artist_tracks = df['Track_Uri'].tolist()

    return top_artist_tracks[0:25]



def main():
    st.title("Spotify Song Recommendations")
    url = st.text_input("Enter the Spotify URL of a song to use as a seed for recommendations:")
    model = st.selectbox("Select Model", ["Spotify Model", "My Model"])

    if st.button("Submit"):
        if model == "Spotify Model":
            result= spotifyModel(url, model)
            if result:
                st.write("Recommended songs:")
                for track in result:
                    uri_link = "https://open.spotify.com/embed/track/" + track + "?utm_source=generator&theme=0"
                    components.iframe(uri_link, height=80)
            else:
                st.write("Error: Invalid URL")
        elif model == "My Model":
            # call your custom model function here
            result = songsModel(url, model)
            if result:
                random.shuffle(result)
                st.write("Recommended songs:")
                for track in result:
                    uri_link = "https://open.spotify.com/embed/track/" + track + "?utm_source=generator&theme=0"
                    components.iframe(uri_link, height=80)
            else:
                st.write("Error: Invalid URL")
    
    st.subheader('Audio Features Explanation')
    """
    | Variable | Description |
    | :----: | :---: |
    | Acousticness | A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. |
    | Danceability | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. |
    | Energy | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. |
    | Instrumentalness | Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. |
    | Key | The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1. |
    | Liveness | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. |
    | Loudness | The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db. |
    | Mode | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0. |
    | Speechiness | Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. |
    | Tempo | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. |
    | Time Signature | An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4". |
    | Valence | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). |
    
    Information about features: [here](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)
    """

if __name__ == '__main__':
    main()
