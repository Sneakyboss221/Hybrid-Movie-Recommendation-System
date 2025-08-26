"""
Content-Based Filtering Module for Hybrid Movie Recommendation System

This module implements:
- TF-IDF vectorization of movie metadata
- Cosine similarity computation
- Content-based movie recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple
import joblib
import os

class ContentBasedRecommender:
    """
    Content-based recommendation system using TF-IDF and cosine similarity.
    """
    
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize the content-based recommender.
        
        Args:
            movies_df: DataFrame containing movie information with 'combined_features' column
        """
        self.movies_df = movies_df.copy()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.is_fitted = False
        
    def fit(self, feature_column: str = 'combined_features', max_features: int = 5000):
        """
        Fit the TF-IDF vectorizer and compute similarity matrix.
        
        Args:
            feature_column: Column name containing combined features
            max_features: Maximum number of features for TF-IDF
        """
        print("Fitting content-based recommender...")
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Apply TF-IDF to the combined features
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df[feature_column].fillna(''))
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create a mapping of movie titles to indices
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['clean_title']).drop_duplicates()
        
        self.is_fitted = True
        print(f"Content-based recommender fitted successfully!")
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Similarity matrix shape: {self.cosine_sim.shape}")
    
    def get_recommendations(self, movie_title: str, num_recommendations: int = 10) -> Optional[pd.DataFrame]:
        """
        Get content-based recommendations for a given movie.
        
        Args:
            movie_title: Title of the movie to find recommendations for
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations and similarity scores
        """
        if not self.is_fitted:
            print("Please fit the model first using fit()")
            return None
        
        # Get the index of the movie that matches the title
        try:
            idx = self.indices[movie_title]
        except KeyError:
            print(f"'{movie_title}' not found in the dataset.")
            print("Available movies (sample):")
            sample_titles = self.movies_df['clean_title'].sample(min(5, len(self.movies_df))).tolist()
            for t in sample_titles:
                print(f"- {t}")
            return None
        
        # Get the similarity scores for all movies
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the most similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:num_recommendations+1]
        
        # Get the movie indices and similarity scores
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Return the top movies with their similarity scores
        recommendations = self.movies_df.iloc[movie_indices].copy()
        recommendations['similarity_score'] = similarity_scores
        
        # Add additional information
        recommendations['content_score'] = similarity_scores
        
        return recommendations[['movieId', 'title', 'genres', 'year', 'similarity_score', 'content_score']]
    
    def get_similar_movies_by_genre(self, genre: str, num_recommendations: int = 10) -> pd.DataFrame:
        """
        Get movies similar to a specific genre.
        
        Args:
            genre: Genre to find similar movies for
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations
        """
        # Filter movies by genre
        genre_movies = self.movies_df[self.movies_df['genres'].str.contains(genre, case=False, na=False)]
        
        if len(genre_movies) == 0:
            print(f"No movies found with genre '{genre}'")
            return pd.DataFrame()
        
        # Get the most popular movies in this genre
        popular_genre_movies = genre_movies.sort_values('year', ascending=False).head(num_recommendations)
        
        return popular_genre_movies[['movieId', 'title', 'genres', 'year']]
    
    def get_movie_features(self, movie_title: str) -> Optional[Dict]:
        """
        Get the TF-IDF features for a specific movie.
        
        Args:
            movie_title: Title of the movie
            
        Returns:
            Dictionary with movie features
        """
        if not self.is_fitted:
            print("Please fit the model first using fit()")
            return None
        
        try:
            idx = self.indices[movie_title]
            movie_features = self.tfidf_matrix[idx].toarray()[0]
            
            # Get feature names
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Create dictionary of features and their weights
            features_dict = {
                feature_names[i]: weight 
                for i, weight in enumerate(movie_features) 
                if weight > 0
            }
            
            # Sort by weight
            features_dict = dict(sorted(features_dict.items(), key=lambda x: x[1], reverse=True))
            
            return features_dict
            
        except KeyError:
            print(f"'{movie_title}' not found in the dataset.")
            return None
    
    def get_top_features(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get the top TF-IDF features across all movies.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature, weight) tuples
        """
        if not self.is_fitted:
            print("Please fit the model first using fit()")
            return []
        
        # Calculate average feature weights
        avg_features = np.mean(self.tfidf_matrix.toarray(), axis=0)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Create list of (feature, weight) tuples
        features = list(zip(feature_names, avg_features))
        
        # Sort by weight and return top N
        features.sort(key=lambda x: x[1], reverse=True)
        return features[:top_n]
    
    def explain_recommendation(self, movie_title: str, recommended_movie_title: str) -> Optional[str]:
        """
        Explain why a movie was recommended based on shared features.
        
        Args:
            movie_title: Original movie title
            recommended_movie_title: Recommended movie title
            
        Returns:
            Explanation string
        """
        if not self.is_fitted:
            print("Please fit the model first using fit()")
            return None
        
        try:
            # Get features for both movies
            original_features = self.get_movie_features(movie_title)
            recommended_features = self.get_movie_features(recommended_movie_title)
            
            if original_features is None or recommended_features is None:
                return None
            
            # Find shared features
            shared_features = set(original_features.keys()) & set(recommended_features.keys())
            
            if not shared_features:
                return f"No specific shared features found between '{movie_title}' and '{recommended_movie_title}'"
            
            # Get top shared features
            shared_feature_weights = {
                feature: min(original_features[feature], recommended_features[feature])
                for feature in shared_features
            }
            
            top_shared = sorted(shared_feature_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            
            explanation = f"'{recommended_movie_title}' was recommended because it shares these features with '{movie_title}': "
            explanation += ", ".join([f"'{feature}'" for feature, weight in top_shared])
            
            return explanation
            
        except Exception as e:
            print(f"Error explaining recommendation: {e}")
            return None
    
    def save_model(self, filepath: str):
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            print("No fitted model to save")
            return
        
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'cosine_sim': self.cosine_sim,
            'indices': self.indices,
            'movies_df': self.movies_df
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a fitted model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return
        
        model_data = joblib.load(filepath)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.cosine_sim = model_data['cosine_sim']
        self.indices = model_data['indices']
        self.movies_df = model_data['movies_df']
        self.is_fitted = True
        
        print(f"Model loaded from {filepath}")
    
    def get_recommendation_diversity(self, movie_title: str, num_recommendations: int = 10) -> float:
        """
        Calculate diversity of recommendations for a movie.
        
        Args:
            movie_title: Title of the movie
            num_recommendations: Number of recommendations to consider
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        recommendations = self.get_recommendations(movie_title, num_recommendations)
        
        if recommendations is None or len(recommendations) == 0:
            return 0.0
        
        # Calculate average similarity between recommended movies
        if len(recommendations) < 2:
            return 1.0
        
        # Get indices of recommended movies
        rec_indices = recommendations.index.tolist()
        
        # Calculate average similarity between recommendations
        similarities = []
        for i in range(len(rec_indices)):
            for j in range(i+1, len(rec_indices)):
                sim = self.cosine_sim[rec_indices[i], rec_indices[j]]
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        
        # Diversity is 1 - average similarity
        diversity = 1 - avg_similarity
        
        return diversity

def main():
    """Example usage of the ContentBasedRecommender class."""
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    cleaned_movies = preprocessor.clean_data()
    
    # Initialize and fit content-based recommender
    content_recommender = ContentBasedRecommender(cleaned_movies)
    content_recommender.fit()
    
    # Get recommendations
    movie_title = "Toy Story"
    recommendations = content_recommender.get_recommendations(movie_title, num_recommendations=5)
    
    if recommendations is not None:
        print(f"\nContent-based recommendations for '{movie_title}':")
        print(recommendations[['title', 'genres', 'similarity_score']])
        
        # Explain a recommendation
        if len(recommendations) > 0:
            explanation = content_recommender.explain_recommendation(
                movie_title, 
                recommendations.iloc[0]['title']
            )
            print(f"\nExplanation: {explanation}")
        
        # Get diversity score
        diversity = content_recommender.get_recommendation_diversity(movie_title)
        print(f"\nRecommendation diversity: {diversity:.3f}")

if __name__ == "__main__":
    main()