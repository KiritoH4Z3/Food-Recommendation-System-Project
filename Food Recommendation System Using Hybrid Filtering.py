import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize


# Function to load and merge data
def load_and_merge_data(food_file_path, ratings_file_path):
    # Load the datasets
    food_df = pd.read_csv(food_file_path)
    ratings_df = pd.read_csv(ratings_file_path)

    # Merge the ratings data with the food data
    return pd.merge(food_df, ratings_df, on='Food_ID')


# Function for data preprocessing with tokenization
def preprocess_data(df):
    # Data preprocessing with tokenization
    df["Describe"] = df["Describe"].apply(lambda x: word_tokenize(x))
    df["C_Type"] = df["C_Type"].apply(lambda x: word_tokenize(x))
    df["Name"] = df["Name"].apply(lambda x: word_tokenize(x))
    df["Veg_Non"] = df["Veg_Non"].apply(lambda x: word_tokenize(x))
    df["Tags"] = df["Name"] + df["C_Type"] + df["Veg_Non"] + df["Describe"]

    # Create a new dataframe
    new_df = df[["Food_ID", "Name", "Tags"]].copy()

    # Further preprocessing
    new_df["Tags"] = new_df["Tags"].apply(lambda x: " ".join(x))
    new_df["Tags"] = new_df["Tags"].apply(lambda x: x.lower())

    return new_df


# Function to calculate cosine similarity
def calculate_similarity(df):
    # Feature extraction with TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", max_features=500)
    vector = tfidf.fit_transform(df["Tags"]).toarray()

    # Calculate cosine similarity
    return cosine_similarity(vector)


# Function for recommendation
def recommender(food, new_df, sim, df):
    food_df = new_df[new_df["Tags"].apply(lambda x: food in x.split())]
    cosine_values = ""
    recommendation_results = ""

    if not food_df.empty:
        food_index = food_df.index[0]
        distances = sim[food_index]
        food_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:16]

        # Get the Food_IDs of the recommended foods
        recommended_food_ids = [new_df.iloc[i[0]]['Food_ID'] for i in food_list]

        # Get the average ratings of the recommended foods
        recommended_foods = df[df['Food_ID'].isin(recommended_food_ids)]
        average_ratings = recommended_foods.groupby('Food_ID')['Rating'].mean()

        # Sort the recommended foods by their average rating
        sorted_food_list = average_ratings.sort_values(ascending=False).index.tolist()

        recommendation_results += "Here is a list of dishes with " + food + ", sorted by average rating:\n"
        for i, food_id in enumerate(sorted_food_list, start=1):
            recommendation_results += f"{i}. {' '.join(new_df[new_df['Food_ID'] == food_id]['Name'].values[0])}\n"
            cosine_values += f"Food_ID: {food_id}, Cosine Similarity: {distances[i - 1]}\n"

        tokenized_tags = "\nTokenized Tags:\n" + "\n".join(df["Tags"].apply(lambda x: " ".join(x)))
        return "Recommendation generated.", cosine_values, tokenized_tags, recommendation_results
    else:
        return "Food not found.", cosine_values, "", ""


# Main function
def main():
    # Define the file path for your dataset
    food_file_path = r"C:\Users\silen\OneDrive\Desktop\YEAR 2\AI\food.csv"
    ratings_file_path = r"C:\Users\silen\OneDrive\Desktop\YEAR 2\AI\ratings.csv"

    # Load and preprocess data
    global df, new_df, sim
    df = load_and_merge_data(food_file_path, ratings_file_path)
    new_df = preprocess_data(df)
    sim = calculate_similarity(new_df)

    # Console input loop
    while True:
        user_input = input("User: ")

        if user_input.lower() == "exit":  # Check if user wants to exit
            print("Chatbot: Goodbye!")
            break  # Exit the loop and end the program

        greetings = ["hello", "hi", "greetings", "hey"]
        if any(word.lower() in greetings for word in user_input.split()):
            response = "Hello! What type of food would you like today?"
        else:
            response, cosine_values, tokenized_tags, recommendation_results = recommender(
                user_input.lower(), new_df, sim, df
            )

            print(tokenized_tags)
            print("\nCosine Similarity Values:")
            print(cosine_values)
            print("\nRecommendation Ranking Results:")
            print(recommendation_results)

        print("Chatbot: " + response)


if __name__ == "__main__":
    main()
