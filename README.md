# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Python, Natural Language Processing (NLP), Machine Learning, and Flask.

The application recommends movies similar to a movie entered by the user.

---

## 📌 Project Overview

This project uses the TMDB 5000 Movies dataset to build a movie recommendation system.

The system analyzes movie information such as:

- Movie overview
- Genres
- Keywords
- Top 3 cast members
- Director

These features are combined into a single set of tags and processed using NLP techniques.

The system then uses TF-IDF vectorization and cosine similarity to find movies that are most similar to the selected movie.

---

## 🚀 Features

- 🔎 Search for a movie by title
- 🎬 Get the top 5 similar movies
- 🤖 Content-based recommendation
- 🧠 NLP text preprocessing
- 📊 TF-IDF vectorization
- 📐 Cosine similarity
- 🌐 Flask web application
- 📱 Responsive web interface
- ⭐ Similarity scores for recommendations

---

## 🧠 How the Recommendation System Works

The recommendation pipeline is:

```text
TMDB Movie Dataset
        ↓
Data Cleaning
        ↓
Feature Selection
        ↓
Genres + Keywords + Overview
+ Cast + Director
        ↓
Create Movie Tags
        ↓
Lowercase Text
        ↓
Porter Stemming
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Top 5 Recommendations
