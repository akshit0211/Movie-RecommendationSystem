# Movie Recommendation System

This repository contains a movie recommendation system built with Docker, Bitnami/Kafka, Bitnami/Spark, and TensorFlow. The system uses the MovieLens dataset to generate movie recommendations for users.

## Project Overview

The recommendation system works by taking in user ratings for movies, learning the user's preferences over time, and suggesting movies that the user might like. The project uses Spark to process the data, Kafka to handle real-time data streams, and TensorFlow for machine learning tasks.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed Docker.
- You have installed Kubernetes. For local development, you can use Docker Desktop which includes a standalone Kubernetes server and client. 

## Installing Movie Recommendation System

To install the Movie Recommendation System, follow these steps:

```bash
git clone https://github.com/akshit0211/Movie-RecommendationSystem.git
cd Movie-RecommendationSystem

kubectl apply -f kafkadeployment.yaml
kubectl apply -f kafkaservice.yaml
--
kubectl apply -f sparkdeployment.yaml
kubectl apply -f sparkservice.yaml
--
kubectl apply -f tensorflowdeployment.yaml
kubectl apply -f tensorflowservice.yaml
