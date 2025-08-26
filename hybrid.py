"""
Hybrid Recommendation Module for Hybrid Movie Recommendation System

This module implements:
- Weighted combination of content-based and collaborative filtering
- Cold-start handling for new users and movies
- Ensemble recommendations
- Serendipitous recommendations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
import warnings
warnings.filterwarnings('ignore')

class HybridRecommender:
    """
    Hybrid recommendation system combining content-based and collaborative filtering.
    """
    
    def __init__(self, content_recommender: ContentBasedRecommender, 
                 collaborative_recommender: CollaborativeRecommender):
        """
        Initialize the hybrid recommender.
        
        Args:
            content_recommender: Fitted content-based recommender
            collaborative_recommender: Fitted collaborative recommender
        """
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.movies_df = content_recommender.movies_df
        self.ratings_df = collaborative_recommender.ratings_df
        
        # Create movie ID mappings
        self.movie_id_to_title = collaborative_recommender.movie_id_to_title
        self.title_to_movie_id = collaborative_recommender.title_to_movie_id
        
    def get_hybrid_recommendations(self, movie_title: str = None, user_id: int = None, 
                                 num_recommendations: int = 10, alpha: float = 0.5) -> Optional[pd.DataFrame]:
        """
        Get hybrid recommendations combining content-based and collaborative filtering.
        
        Args:
            movie_title: Title of the movie (for content-based)
            user_id: ID of the user (for collaborative)
            num_recommendations: Number of recommendations to return
            alpha: Weight for content-based filtering (0-1). 
                  0 = only collaborative, 1 = only content-based
                  
        Returns:
            DataFrame with hybrid recommendations
        """
        if movie_title is None and user_id is None:
            print("Please provide either movie_title or user_id")
            return None
        
        # Handle cold-start scenarios
        if movie_title and user_id is None:
            # Content-based only (new user scenario)
            return self._get_content_based_only(movie_title, num_recommendations)
        
        elif user_id and movie_title is None:
            # Collaborative only (new movie scenario)
            return self._get_collaborative_only(user_id, num_recommendations)
        
        else:
            # Hybrid approach
            return self._get_hybrid_combination(movie_title, user_id, num_recommendations, alpha)
    
    def _get_content_based_only(self, movie_title: str, num_recommendations: int) -> Optional[pd.DataFrame]:
        """
        Get content-based recommendations only (for new users).
        
        Args:
            movie_title: Title of the movie
            num_recommendations: Number of recommendations
            
        Returns:
            DataFrame with content-based recommendations
        """
        recommendations = self.content_recommender.get_recommendations(movie_title, num_recommendations)
        
        if recommendations is not None:
            recommendations['hybrid_score'] = recommendations['content_score']
            recommendations['recommendation_type'] = 'content_based'
        
        return recommendations
    
    def _get_collaborative_only(self, user_id: int, num_recommendations: int) -> Optional[pd.DataFrame]:
        """
        Get collaborative recommendations only (for new movies).
        
        Args:
            user_id: ID of the user
            num_recommendations: Number of recommendations
            
        Returns:
            DataFrame with collaborative recommendations
        """
        recommendations = self.collaborative_recommender.get_user_recommendations(user_id, num_recommendations)
        
        if recommendations is not None:
            recommendations['hybrid_score'] = recommendations['collaborative_score']
            recommendations['recommendation_type'] = 'collaborative'
        
        return recommendations
    
    def _get_hybrid_combination(self, movie_title: str, user_id: int, 
                              num_recommendations: int, alpha: float) -> Optional[pd.DataFrame]:
        """
        Combine content-based and collaborative recommendations.
        
        Args:
            movie_title: Title of the movie
            user_id: ID of the user
            num_recommendations: Number of recommendations
            alpha: Weight for content-based filtering
            
        Returns:
            DataFrame with hybrid recommendations
        """
        # Get content-based recommendations
        content_recs = self.content_recommender.get_recommendations(movie_title, num_recommendations * 2)
        
        # Get collaborative recommendations
        collab_recs = self.collaborative_recommender.get_user_recommendations(user_id, num_recommendations * 2)
        
        if content_recs is None or collab_recs is None:
            print("Could not get recommendations from one or both models")
            return None
        
        # Create a combined dataset
        combined_recs = self._combine_recommendations(content_recs, collab_recs, alpha)
        
        # Sort by hybrid score and return top N
        combined_recs = combined_recs.sort_values('hybrid_score', ascending=False).head(num_recommendations)
        
        return combined_recs
    
    def _combine_recommendations(self, content_recs: pd.DataFrame, collab_recs: pd.DataFrame, 
                               alpha: float) -> pd.DataFrame:
        """
        Combine content-based and collaborative recommendations with weighted scoring.
        
        Args:
            content_recs: Content-based recommendations
            collab_recs: Collaborative recommendations
            alpha: Weight for content-based filtering
            
        Returns:
            Combined recommendations DataFrame
        """
        # Normalize scores to 0-1 range
        content_recs['content_score_norm'] = (content_recs['content_score'] - content_recs['content_score'].min()) / \
                                           (content_recs['content_score'].max() - content_recs['content_score'].min())
        
        collab_recs['collaborative_score_norm'] = (collab_recs['collaborative_score'] - collab_recs['collaborative_score'].min()) / \
                                                (collab_recs['collaborative_score'].max() - collab_recs['collaborative_score'].min())
        
        # Create a combined dataset
        all_movies = set(content_recs['movieId'].tolist()) | set(collab_recs['movieId'].tolist())
        
        combined_data = []
        for movie_id in all_movies:
            # Get content-based score
            content_row = content_recs[content_recs['movieId'] == movie_id]
            content_score = content_row['content_score_norm'].iloc[0] if not content_row.empty else 0
            
            # Get collaborative score
            collab_row = collab_recs[collab_recs['movieId'] == movie_id]
            collab_score = collab_row['collaborative_score_norm'].iloc[0] if not collab_row.empty else 0
            
            # Calculate hybrid score
            hybrid_score = alpha * content_score + (1 - alpha) * collab_score
            
            # Get movie information
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                genres = movie_info.iloc[0]['genres']
                year = movie_info.iloc[0].get('year', 0)
            else:
                title = f"Unknown ({movie_id})"
                genres = "Unknown"
                year = 0
            
            combined_data.append({
                'movieId': movie_id,
                'title': title,
                'genres': genres,
                'year': year,
                'content_score': content_score,
                'collaborative_score': collab_score,
                'hybrid_score': hybrid_score,
                'recommendation_type': 'hybrid'
            })
        
        return pd.DataFrame(combined_data)
    
    def get_ensemble_recommendations(self, movie_title: str = None, user_id: int = None,
                                   num_recommendations: int = 10, 
                                   weights: Dict[str, float] = None) -> Optional[pd.DataFrame]:
        """
        Get ensemble recommendations using multiple algorithms.
        
        Args:
            movie_title: Title of the movie
            user_id: ID of the user
            num_recommendations: Number of recommendations
            weights: Dictionary with weights for each algorithm
                    {'content': 0.4, 'collaborative': 0.4, 'popularity': 0.2}
            
        Returns:
            DataFrame with ensemble recommendations
        """
        if weights is None:
            weights = {'content': 0.4, 'collaborative': 0.4, 'popularity': 0.2}
        
        # Get recommendations from different algorithms
        recommendations = {}
        
        if movie_title:
            content_recs = self.content_recommender.get_recommendations(movie_title, num_recommendations * 2)
            if content_recs is not None:
                recommendations['content'] = content_recs
        
        if user_id:
            collab_recs = self.collaborative_recommender.get_user_recommendations(user_id, num_recommendations * 2)
            if collab_recs is not None:
                recommendations['collaborative'] = collab_recs
        
        # Get popularity-based recommendations
        popularity_recs = self._get_popularity_recommendations(num_recommendations * 2)
        if popularity_recs is not None:
            recommendations['popularity'] = popularity_recs
        
        # Combine all recommendations
        combined_recs = self._combine_ensemble_recommendations(recommendations, weights)
        
        # Sort by ensemble score and return top N
        combined_recs = combined_recs.sort_values('ensemble_score', ascending=False).head(num_recommendations)
        
        return combined_recs
    
    def _get_popularity_recommendations(self, num_recommendations: int) -> pd.DataFrame:
        """
        Get popularity-based recommendations.
        
        Args:
            num_recommendations: Number of recommendations
            
        Returns:
            DataFrame with popularity-based recommendations
        """
        # Calculate movie popularity based on number of ratings and average rating
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']
        
        # Calculate popularity score (normalized)
        movie_stats['rating_count_norm'] = (movie_stats['rating_count'] - movie_stats['rating_count'].min()) / \
                                         (movie_stats['rating_count'].max() - movie_stats['rating_count'].min())
        
        movie_stats['rating_mean_norm'] = (movie_stats['rating_mean'] - movie_stats['rating_mean'].min()) / \
                                        (movie_stats['rating_mean'].max() - movie_stats['rating_mean'].min())
        
        movie_stats['popularity_score'] = 0.7 * movie_stats['rating_count_norm'] + 0.3 * movie_stats['rating_mean_norm']
        
        # Get top popular movies
        popular_movies = movie_stats.sort_values('popularity_score', ascending=False).head(num_recommendations)
        
        # Add movie information
        popular_movies = popular_movies.merge(self.movies_df[['movieId', 'title', 'genres', 'year']], 
                                            on='movieId', how='left')
        
        return popular_movies[['movieId', 'title', 'genres', 'year', 'popularity_score']]
    
    def _combine_ensemble_recommendations(self, recommendations: Dict[str, pd.DataFrame], 
                                        weights: Dict[str, float]) -> pd.DataFrame:
        """
        Combine recommendations from multiple algorithms.
        
        Args:
            recommendations: Dictionary of recommendation DataFrames
            weights: Dictionary of weights for each algorithm
            
        Returns:
            Combined recommendations DataFrame
        """
        all_movies = set()
        for recs in recommendations.values():
            all_movies.update(recs['movieId'].tolist())
        
        combined_data = []
        for movie_id in all_movies:
            # Get scores from each algorithm
            scores = {}
            for algo, recs in recommendations.items():
                movie_row = recs[recs['movieId'] == movie_id]
                if not movie_row.empty:
                    if algo == 'content':
                        scores[algo] = movie_row['content_score'].iloc[0]
                    elif algo == 'collaborative':
                        scores[algo] = movie_row['collaborative_score'].iloc[0]
                    elif algo == 'popularity':
                        scores[algo] = movie_row['popularity_score'].iloc[0]
                else:
                    scores[algo] = 0
            
            # Normalize scores
            for algo in scores:
                if algo in recommendations:
                    algo_recs = recommendations[algo]
                    if algo == 'content':
                        max_score = algo_recs['content_score'].max()
                        min_score = algo_recs['content_score'].min()
                    elif algo == 'collaborative':
                        max_score = algo_recs['collaborative_score'].max()
                        min_score = algo_recs['collaborative_score'].min()
                    elif algo == 'popularity':
                        max_score = algo_recs['popularity_score'].max()
                        min_score = algo_recs['popularity_score'].min()
                    
                    if max_score > min_score:
                        scores[algo] = (scores[algo] - min_score) / (max_score - min_score)
                    else:
                        scores[algo] = 0
            
            # Calculate ensemble score
            ensemble_score = sum(weights.get(algo, 0) * scores.get(algo, 0) for algo in weights)
            
            # Get movie information
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                genres = movie_info.iloc[0]['genres']
                year = movie_info.iloc[0].get('year', 0)
            else:
                title = f"Unknown ({movie_id})"
                genres = "Unknown"
                year = 0
            
            combined_data.append({
                'movieId': movie_id,
                'title': title,
                'genres': genres,
                'year': year,
                'ensemble_score': ensemble_score,
                'content_score': scores.get('content', 0),
                'collaborative_score': scores.get('collaborative', 0),
                'popularity_score': scores.get('popularity', 0),
                'recommendation_type': 'ensemble'
            })
        
        return pd.DataFrame(combined_data)
    

    
    def get_recommendation_diversity(self, recommendations: pd.DataFrame) -> float:
        """
        Calculate diversity of recommendations.
        
        Args:
            recommendations: DataFrame with recommendations
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(recommendations) < 2:
            return 1.0
        
        # Calculate genre diversity
        all_genres = []
        for genres in recommendations['genres']:
            all_genres.extend(genres.split('|'))
        
        unique_genres = set(all_genres)
        genre_diversity = len(unique_genres) / len(all_genres) if all_genres else 0
        
        # Calculate year diversity
        years = recommendations['year'].dropna()
        if len(years) > 1:
            year_diversity = 1 - (years.std() / years.mean()) if years.mean() > 0 else 0
        else:
            year_diversity = 0
        
        # Calculate score diversity
        scores = recommendations['hybrid_score'] if 'hybrid_score' in recommendations.columns else recommendations.get('ensemble_score', pd.Series([0]))
        if len(scores) > 1:
            score_diversity = 1 - (scores.std() / scores.mean()) if scores.mean() > 0 else 0
        else:
            score_diversity = 0
        
        # Combine diversities
        overall_diversity = (genre_diversity + year_diversity + score_diversity) / 3
        
        return overall_diversity
    
    def get_serendipitous_recommendations(self, movie_title: str, user_id: int,
                                        num_recommendations: int = 10, 
                                        serendipity_weight: float = 0.3) -> Optional[pd.DataFrame]:
        """
        Get serendipitous recommendations that balance similarity and novelty.
        
        Args:
            movie_title: Title of the movie
            user_id: ID of the user
            num_recommendations: Number of recommendations
            serendipity_weight: Weight for serendipity (0-1)
            
        Returns:
            DataFrame with serendipitous recommendations
        """
        # Get regular hybrid recommendations
        regular_recs = self.get_hybrid_recommendations(movie_title, user_id, num_recommendations * 2)
        
        if regular_recs is None:
            return None
        
        # Calculate serendipity scores
        serendipity_scores = []
        for _, row in regular_recs.iterrows():
            movie_id = row['movieId']
            
            # Get movie popularity (inverse for serendipity)
            movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]
            popularity = len(movie_ratings) / len(self.ratings_df)
            serendipity = 1 - popularity
            
            # Get genre novelty (how different from user's typical preferences)
            if user_id:
                user_profile = self.collaborative_recommender.get_user_profile(user_id)
                if user_profile and 'favorite_genres' in user_profile:
                    favorite_genres = set([genre for genre, _ in user_profile['favorite_genres']])
                    movie_genres = set(row['genres'].split('|'))
                    genre_novelty = 1 - len(favorite_genres & movie_genres) / len(favorite_genres) if favorite_genres else 0
                    serendipity = (serendipity + genre_novelty) / 2
            
            serendipity_scores.append(serendipity)
        
        regular_recs['serendipity_score'] = serendipity_scores
        
        # Combine similarity and serendipity
        regular_recs['final_score'] = (1 - serendipity_weight) * regular_recs['hybrid_score'] + \
                                    serendipity_weight * regular_recs['serendipity_score']
        
        # Sort by final score and return top N
        serendipitous_recs = regular_recs.sort_values('final_score', ascending=False).head(num_recommendations)
        
        return serendipitous_recs

def main():
    """Example usage of the HybridRecommender class."""
    from data_preprocessing import DataPreprocessor
    from content_based import ContentBasedRecommender
    from collaborative import CollaborativeRecommender
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    cleaned_movies = preprocessor.clean_data()
    merged_data = preprocessor.merge_datasets()
    
    # Initialize recommenders
    content_recommender = ContentBasedRecommender(cleaned_movies)
    content_recommender.fit()
    
    collab_recommender = CollaborativeRecommender(data['ratings'], cleaned_movies)
    collab_recommender.prepare_data()
    collab_recommender.fit_svd()
    
    # Initialize hybrid recommender
    hybrid_recommender = HybridRecommender(content_recommender, collab_recommender)
    
    # Get hybrid recommendations
    movie_title = "Toy Story"
    user_id = 1
    alpha = 0.6  # 60% content-based, 40% collaborative
    
    hybrid_recs = hybrid_recommender.get_hybrid_recommendations(
        movie_title=movie_title,
        user_id=user_id,
        num_recommendations=5,
        alpha=alpha
    )
    
    if hybrid_recs is not None:
        print(f"\nHybrid recommendations (alpha={alpha}):")
        print(hybrid_recs[['title', 'genres', 'hybrid_score']])
        

        
        # Get diversity score
        diversity = hybrid_recommender.get_recommendation_diversity(hybrid_recs)
        print(f"\nRecommendation diversity: {diversity:.3f}")

if __name__ == "__main__":
    main()
