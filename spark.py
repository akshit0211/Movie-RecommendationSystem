from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

# Defining the schema to parse the JSON data
ratings_schema = StructType([
    StructField("userId", IntegerType()),
    StructField("movieId", IntegerType()),
    StructField("rating", FloatType()),
])

#Creating a SparkSession
spark = SparkSession.builder.appName("MovieRatingsProcessor").config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1").getOrCreate() 

#Reading the movie ratings data from the Kafka topic
df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "kafka_container:9092").option("subscribe", "movie_ratings").load()

#Converting the binary value column to string
df = df.selectExpr("CAST(value AS STRING)")

#The data is in JSON format, so we parse it
df = df.select(F.from_json(F.col("value").cast("string"), ratings_schema).alias("parsed_value"))

#Selecting the values from the parsed JSON
df = df.select(
    F.col("parsed_value.userId").alias("userId"), 
    F.col("parsed_value.movieId").alias("movieId"),
    F.col("parsed_value.rating").alias("rating")
)

#Printing the schema of the DataFrame
# df.printSchema()
# df = df.na.drop(subset=["userId"])

#Converting user_id and movie_id to integer indices
indexer = StringIndexer(inputCols=["userId", "movieId"], outputCols=["userId_index", "movieId_index"])
pipeline = Pipeline(stages=[indexer])

def process_batch(df, epoch_id):
    transformed_df = pipeline.fit(df).transform(df)
    transformed_df.write.csv("/home/jovyan/work/data/data_" + str(epoch_id), mode="overwrite")

#Starting streaming query
query = df \
    .writeStream \
    .outputMode("append") \
    .foreachBatch(process_batch) \
    .start()
# query = df \
#     .writeStream \
#     .outputMode("update") \
#     .format("console") \
#     .start()


#Waiting for the streaming query to finish
query.awaitTermination()

#Stopping the SparkSession
spark.stop()