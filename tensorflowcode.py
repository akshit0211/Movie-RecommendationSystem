import glob
import tensorflow as tf
import pandas as pd
import numpy as np

all_csv_files = glob.glob('/home/jovyan/work/data/**/*.csv', recursive=True)

datasets = []

for csv_file in all_csv_files:
    dataset = tf.data.experimental.CsvDataset(csv_file, [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
    datasets.append(dataset)

data = []
for dataset in datasets:
    for element in dataset.take(5):
        x = ([item.numpy() for item in element])
        data.append([int(x[0]), int(x[1]), int(x[2])])
        
df = pd.DataFrame(data, columns=['user_id', 'movie_id', 'rating'])

user_ids = df['user_id'].values.astype('int32')
movie_ids = df['movie_id'].values.astype('int32')
ratings = df['rating'].values.astype('int32')

# Model
max_user_id = df['user_id'].max()
max_movie_id = df['movie_id'].max()

class NeuralCollaborativeFiltering(tf.keras.Model):
    def __init__(self):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(max_user_id + 1, 64)
        self.movie_embedding = tf.keras.layers.Embedding(max_movie_id + 1, 64)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        user_id, movie_id = inputs
        user_emb = self.user_embedding(user_id)
        movie_emb = self.movie_embedding(movie_id)
        x = tf.concat([user_emb, movie_emb], axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

model = NeuralCollaborativeFiltering()
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())

# Train
model.fit([user_ids, movie_ids], ratings, epochs=50, verbose=2)

# Recommendations
user_id = np.array([0])
movie_ids = np.arange(df['movie_id'].nunique())
predictions = model.predict([np.repeat(user_id, df['movie_id'].nunique()), movie_ids])
top_movie_id = np.argmax(predictions)
print(f"Recommended movie for user {user_id[0]}: {top_movie_id}")
