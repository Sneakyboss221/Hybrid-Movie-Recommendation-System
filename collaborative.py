"""
Collaborative Filtering Module for Hybrid Movie Recommendation System

This module implements:
- SVD (Singular Value Decomposition) using Surprise library
- NMF (Non-negative Matrix Factorization) using Surprise library
- User-based collaborative filtering recommendations
- Model evaluation and comparison
"""

import pandas as pd
import numpy as np
from surprise import SVD, NMF, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    """
    Collaborative filtering recommendation system using Surprise library.
    """
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Initialize the collaborative filtering recommender.
        
        Args:
            ratings_df: DataFrame with columns ['userId', 'movieId', 'rating']
            movies_df: DataFrame with movie information
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        self.svd_model = None
        self.nmf_model = None
        self.trainset = None
        self.testset = None
        self.is_fitted = False
        self.current_model = None
        
        # Create movie ID to title mapping
        self.movie_id_to_title = pd.Series(
            movies_df['title'].values, 
            index=movies_df['movieId']
        ).to_dict()
        
        # Create title to movie ID mapping
        self.title_to_movie_id = pd.Series(
            movies_df['movieId'].values, 
            index=movies_df['clean_title']
        ).to_dict()
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Prepare data for Surprise library format.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        print("Preparing data for collaborative filtering...")
        
        # Define the rating scale
        reader = Reader(rating_scale=(1, 5))
        
        # Load the dataset
        data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Split the data
        self.trainset, self.testset = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"Training set size: {self.trainset.n_ratings}")
        print(f"Test set size: {len(self.testset)}")
    
    def fit_svd(self, n_factors: int = 100, n_epochs: int = 20, lr_all: float = 0.005, reg_all: float = 0.02):
        """
        Fit SVD model.
        
        Args:
            n_factors: Number of latent factors
            n_epochs: Number of iterations for SGD
            lr_all: Learning rate for all parameters
            reg_all: Regularization term for all parameters
        """
        print("Fitting SVD model...")
        
        if self.trainset is None:
            print("Please prepare data first using prepare_data()")
            return
        
        self.svd_model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        
        self.svd_model.fit(self.trainset)
        self.current_model = self.svd_model
        self.is_fitted = True
        
        print("SVD model fitted successfully!")
    
    def fit_nmf(self, n_factors: int = 100, n_epochs: int = 50, random_state: int = 42):
        """
        Fit NMF model.
        
        Args:
            n_factors: Number of latent factors
            n_epochs: Number of iterations
            random_state: Random seed
        """
        print("Fitting NMF model...")
        
        if self.trainset is None:
            print("Please prepare data first using prepare_data()")
            return
        
        self.nmf_model = NMF(
            n_factors=n_factors,
            n_epochs=n_epochs,
            random_state=random_state
        )
        
        self.nmf_model.fit(self.trainset)
        self.current_model = self.nmf_model
        self.is_fitted = True
        
        print("NMF model fitted successfully!")
    
    def evaluate_model(self, model_name: str = None) -> Dict[str, float]:
        """
        Evaluate the current model on test set.
        
        Args:
            model_name: Name of the model to evaluate ('svd' or 'nmf')
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            print("Please fit a model first")
            return {}
        
        if model_name:
            if model_name.lower() == 'svd' and self.svd_model:
                model = self.svd_model
            elif model_name.lower() == 'nmf' and self.nmf_model:
                model = self.nmf_model
            else:
                print(f"Model '{model_name}' not available")
                return {}
        else:
            model = self.current_model
        
        if model is None:
            print("No model available for evaluation")
            return {}
        
        # Make predictions on test set
        predictions = model.test(self.testset)
        
        # Calculate metrics
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        metrics = {
            'rmse': rmse,
            'mae': mae
        }
        
        print(f"Model Evaluation:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return metrics
    
    def cross_validate_model(self, model_name: str = 'svd', cv: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the model.
        
        Args:
            model_name: Name of the model ('svd' or 'nmf')
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        if model_name.lower() == 'svd':
            model = SVD(random_state=42)
        elif model_name.lower() == 'nmf':
            model = NMF(random_state=42)
        else:
            print(f"Unknown model: {model_name}")
            return {}
        
        # Load full dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, 
            data, 
            measures=['RMSE', 'MAE'], 
            cv=cv, 
            verbose=True
        )
        
        return cv_results
    
    def get_user_recommendations(self, user_id: int, num_recommendations: int = 10) -> Optional[pd.DataFrame]:
        """
        Get movie recommendations for a specific user.
        
        Args:
            user_id: ID of the user
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations and predicted ratings
        """
        if not self.is_fitted:
            print("Please fit a model first")
            return None
        
        if self.current_model is None:
            print("No model available")
            return None
        
        # Get movies the user hasn't rated
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        rated_movies = set(user_ratings['movieId'].tolist())
        
        # Get all movies
        all_movies = set(self.movies_df['movieId'].tolist())
        unrated_movies = all_movies - rated_movies
        
        if not unrated_movies:
            print(f"User {user_id} has rated all available movies")
            return None
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.current_model.predict(user_id, movie_id).est
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_predictions = predictions[:num_recommendations]
        
        # Create recommendations DataFrame
        recommendations_data = []
        for movie_id, predicted_rating in top_predictions:
            movie_title = self.movie_id_to_title.get(movie_id, f"Unknown ({movie_id})")
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            
            if not movie_info.empty:
                genres = movie_info.iloc[0]['genres']
                year = movie_info.iloc[0].get('year', 0)
            else:
                genres = "Unknown"
                year = 0
            
            recommendations_data.append({
                'movieId': movie_id,
                'title': movie_title,
                'genres': genres,
                'year': year,
                'predicted_rating': predicted_rating,
                'collaborative_score': predicted_rating
            })
        
        recommendations = pd.DataFrame(recommendations_data)
        return recommendations
    
    def get_movie_recommendations(self, movie_title: str, num_recommendations: int = 10) -> Optional[pd.DataFrame]:
        """
        Get similar movies based on collaborative filtering.
        
        Args:
            movie_title: Title of the movie
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with similar movies
        """
        if not self.is_fitted:
            print("Please fit a model first")
            return None
        
        # Get movie ID
        movie_id = self.title_to_movie_id.get(movie_title)
        if movie_id is None:
            print(f"Movie '{movie_title}' not found")
            return None
        
        # Get users who rated this movie
        movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]
        
        if len(movie_ratings) == 0:
            print(f"No ratings found for '{movie_title}'")
            return None
        
        # Get similar movies based on user preferences
        similar_movies = defaultdict(float)
        
        for _, rating_row in movie_ratings.iterrows():
            user_id = rating_row['userId']
            rating = rating_row['rating']
            
            # Get user's other ratings
            user_ratings = self.ratings_df[
                (self.ratings_df['userId'] == user_id) & 
                (self.ratings_df['movieId'] != movie_id)
            ]
            
            for _, other_rating in user_ratings.iterrows():
                other_movie_id = other_rating['movieId']
                other_rating_value = other_rating['rating']
                
                # Calculate similarity based on rating correlation
                similarity = 1 - abs(rating - other_rating_value) / 4.0
                similar_movies[other_movie_id] += similarity
        
        # Sort by similarity score
        sorted_movies = sorted(similar_movies.items(), key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_movies = sorted_movies[:num_recommendations]
        
        # Create recommendations DataFrame
        recommendations_data = []
        for movie_id, similarity_score in top_movies:
            movie_title = self.movie_id_to_title.get(movie_id, f"Unknown ({movie_id})")
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            
            if not movie_info.empty:
                genres = movie_info.iloc[0]['genres']
                year = movie_info.iloc[0].get('year', 0)
            else:
                genres = "Unknown"
                year = 0
            
            recommendations_data.append({
                'movieId': movie_id,
                'title': movie_title,
                'genres': genres,
                'year': year,
                'similarity_score': similarity_score,
                'collaborative_score': similarity_score
            })
        
        recommendations = pd.DataFrame(recommendations_data)
        return recommendations
    
    def get_user_profile(self, user_id: int) -> Dict:
        """
        Get user profile information.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with user profile information
        """
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return {}
        
        profile = {
            'user_id': user_id,
            'total_ratings': len(user_ratings),
            'avg_rating': user_ratings['rating'].mean(),
            'rating_std': user_ratings['rating'].std(),
            'favorite_genres': self._get_favorite_genres(user_ratings),
            'rating_distribution': user_ratings['rating'].value_counts().sort_index().to_dict()
        }
        
        return profile
    
    def _get_favorite_genres(self, user_ratings: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Get user's favorite genres based on ratings.
        
        Args:
            user_ratings: User's ratings DataFrame
            
        Returns:
            List of (genre, average_rating) tuples
        """
        # Merge with movie information
        user_movies = user_ratings.merge(self.movies_df, on='movieId', how='left')
        
        # Explode genres
        genre_ratings = []
        for _, row in user_movies.iterrows():
            genres = row['genres'].split('|')
            for genre in genres:
                genre_ratings.append((genre, row['rating']))
        
        # Calculate average rating per genre
        genre_avg = defaultdict(list)
        for genre, rating in genre_ratings:
            genre_avg[genre].append(rating)
        
        genre_means = [(genre, np.mean(ratings)) for genre, ratings in genre_avg.items()]
        genre_means.sort(key=lambda x: x[1], reverse=True)
        
        return genre_means[:5]  # Top 5 genres
    
    def save_model(self, filepath: str, model_name: str = 'svd'):
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
            model_name: Name of the model to save ('svd' or 'nmf')
        """
        if model_name.lower() == 'svd' and self.svd_model:
            model = self.svd_model
        elif model_name.lower() == 'nmf' and self.nmf_model:
            model = self.nmf_model
        else:
            print(f"Model '{model_name}' not available or not fitted")
            return
        
        model_data = {
            'model': model,
            'trainset': self.trainset,
            'testset': self.testset,
            'ratings_df': self.ratings_df,
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
        
        # Determine which model was loaded
        if hasattr(model_data['model'], 'pu'):
            self.svd_model = model_data['model']
            self.current_model = self.svd_model
        else:
            self.nmf_model = model_data['model']
            self.current_model = self.nmf_model
        
        self.trainset = model_data['trainset']
        self.testset = model_data['testset']
        self.ratings_df = model_data['ratings_df']
        self.movies_df = model_data['movies_df']
        self.is_fitted = True
        
        print(f"Model loaded from {filepath}")

def main():
    """Example usage of the CollaborativeRecommender class."""
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    cleaned_movies = preprocessor.clean_data()
    merged_data = preprocessor.merge_datasets()
    
    # Initialize collaborative recommender
    collab_recommender = CollaborativeRecommender(
        data['ratings'], 
        cleaned_movies
    )
    
    # Prepare data
    collab_recommender.prepare_data()
    
    # Fit SVD model
    collab_recommender.fit_svd()
    
    # Evaluate model
    metrics = collab_recommender.evaluate_model('svd')
    
    # Get user recommendations
    user_id = 1
    user_recommendations = collab_recommender.get_user_recommendations(user_id, num_recommendations=5)
    
    if user_recommendations is not None:
        print(f"\nCollaborative recommendations for user {user_id}:")
        print(user_recommendations[['title', 'genres', 'predicted_rating']])
    
    # Get user profile
    user_profile = collab_recommender.get_user_profile(user_id)
    print(f"\nUser {user_id} profile:")
    for key, value in user_profile.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()