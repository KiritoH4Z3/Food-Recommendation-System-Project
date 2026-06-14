# Food Recommendation System

A hybrid food recommender that combines TF-IDF content similarity with user rating data to suggest dishes from a natural-language food query via a console chatbot.

## Results

- Published in **JATI Journal Vol 8 No 2 (2024)** as sole first author.
- Hybrid approach: **TF-IDF + Collaborative Filtering**.
- Directly addresses the **Cold Start problem** in recommendation systems.
- Achieved a **~75% success rate** in recommendation evaluation.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge&logo=python&logoColor=white)

## Architecture

Food metadata (name, cuisine type, veg/non-veg flag, description) is tokenized with NLTK and merged into a single tag field, then vectorized with `TfidfVectorizer` and compared using cosine similarity to find content-similar dishes. The candidate set from the content model is re-ranked by average user ratings drawn from `ratings.csv`, producing a hybrid content-plus-collaborative ranking. A console loop accepts free-text queries and returns ranked recommendations with their similarity values.

## How to Run

```bash
git clone https://github.com/KiritoH4Z3/Food-Recommendation-System-Project.git
cd Food-Recommendation-System-Project
pip install numpy pandas scikit-learn nltk
python "Food Recommendation System Using Hybrid Filtering.py"
```

The script also requires the NLTK tokenizer data:

```python
import nltk; nltk.download("punkt")
```

Note: update the `food_file_path` and `ratings_file_path` variables in `main()` to point to the included `food.csv` and `ratings.csv`.

## About

Built by Abdullah Mohammed Hazeq as part of peer-reviewed research published in the JATI Journal (Vol 8 No 2, 2024).
