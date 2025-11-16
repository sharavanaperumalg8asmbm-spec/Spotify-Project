# 📦 Import Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 
import spotipy 
from spotipy.oauth2 import SpotifyOAuth 

# 📁 Load Dataset
df = pd.read_csv('spotify_songs.csv')  # Place this CSV in your project folder
print("Initial Data Shape:", df.shape)

# 🧹 Data Preprocessing
df = df.drop(columns=['track_id', 'album_id', 'artist_id'], errors='ignore')
df = df.dropna()

# Normalize numerical features
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 📊 Data Analysis & Visualization
plt.figure(figsize=(10,6))
df['playlist_genre'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation Matrix
plt.figure(figsize=(12,10))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# 🎯 Clustering
X = df[numerical_cols]
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x=df['energy'], y=df['danceability'], hue=df['cluster'], palette='Set2')
plt.title('Clusters by Energy and Danceability')
plt.show()

# 🤖 Recommendation Function
def recommend_songs(genre, cluster_id, n=5):
    subset = df[(df['playlist_genre'] == genre) & (df['cluster'] == cluster_id)]
    return subset.sample(n)[['track_name', 'artist_name']]

# Example usage
print("\n🎵 Recommended Songs:")
print(recommend_songs('pop', 2))

# 🔌 Spotify API Integration
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id='0b13c2235a69452c8661e1fcf62db249',           # Replace with your actual Client ID
    client_secret='d1294c82a6654ad58395d37ac49348ee',   # Replace with your actual Client Secret
    redirect_uri='http://localhost:8888/callback',
    scope='user-library-read playlist-read-private'
))

# Fetch and display user's playlists
print("\n📂 Your Spotify Playlists:")
playlists = sp.current_user_playlists()
for playlist in playlists['items']:
    print(f"- {playlist['name']}")