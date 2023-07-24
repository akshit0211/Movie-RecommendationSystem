from kafka import KafkaProducer
import pandas as pd
import json

try:
    ratings = pd.read_csv('/home/jovyan/work/Datasets/rating.csv')
    # print(ratings.head())
    producer = KafkaProducer(bootstrap_servers='kafka_container:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    for row in ratings.itertuples():
        data = {"userId": row.userId, "movieId": row.movieId, "rating": row.rating}
        producer.send("movie_ratings", data)

    # Ensuring all messages have been delivered
    producer.flush()

except Exception as e:
    print("An error occurred:", e)