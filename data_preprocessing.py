"""
Data Preprocessing Module for Hybrid Movie Recommendation System

This module handles:
- Loading and merging datasets
- Data cleaning and preprocessing
- Feature engineering for content-based filtering
- Optional metadata enrichment via TMDb API
"""

import pandas as pd
import numpy as np
import requests
import json
from typing import Optional, Dict, List
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Handles data preprocessing for the movie recommendation system.
    """
    
    def __init__(self, movies_path: str = 'movies.csv', ratings_path: str = 'ratings.csv'):
        """
        Initialize the data preprocessor.
        
        Args:
            movies_path: Path to movies.csv file
            ratings_path: Path to ratings.csv file
        """
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.movies_df = None
        self.ratings_df = None
        self.merged_df = None
        self.tmdb_api_key = None
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load movies and ratings datasets.
        
        Returns:
            Dictionary containing 'movies' and 'ratings' DataFrames
        """
        print("Loading datasets...")
        
        try:
            self.movies_df = pd.read_csv(self.movies_path)
            self.ratings_df = pd.read_csv(self.ratings_path)
            
            print(f"Movies dataset: {self.movies_df.shape[0]} movies, {self.movies_df.shape[1]} features")
            print(f"Ratings dataset: {self.ratings_df.shape[0]} ratings, {self.ratings_df.shape[1]} features")
            
            return {
                'movies': self.movies_df,
                'ratings': self.ratings_df
            }
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return {}
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the movies dataset.
        
        Returns:
            Cleaned movies DataFrame
        """
        if self.movies_df is None:
            print("Please load data first using load_data()")
            return pd.DataFrame()
        
        print("Cleaning and preprocessing data...")
        
        # Create a copy to avoid modifying original data
        movies_clean = self.movies_df.copy()
        
        # Extract year from title
        movies_clean['year'] = movies_clean['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
        
        # Clean the title by removing the year and ensure string type
        movies_clean['clean_title'] = movies_clean['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True).astype(str)
        
        # Handle missing genres and ensure string type
        movies_clean['genres'] = movies_clean['genres'].fillna('Unknown').astype(str)
        
        # Create combined features for content-based filtering
        movies_clean['combined_features'] = movies_clean['genres'].str.replace('|', ' ')
        
        # Add basic metadata fields (to be enriched later)
        movies_clean['plot'] = ''
        movies_clean['cast'] = ''
        movies_clean['director'] = ''
        movies_clean['keywords'] = ''
        movies_clean['poster_url'] = ''
        
        self.movies_df = movies_clean
        
        print("Data cleaning completed!")
        return movies_clean
    
    def set_tmdb_api_key(self, api_key: str):
        """
        Set TMDb API key for metadata enrichment.
        
        Args:
            api_key: TMDb API key
        """
        self.tmdb_api_key = api_key
        print("TMDb API key set successfully!")
    
    def enrich_metadata(self, max_movies: int = 100) -> pd.DataFrame:
        """
        Enrich movie metadata using TMDb API.
        
        Args:
            max_movies: Maximum number of movies to enrich (for API rate limiting)
            
        Returns:
            Enriched movies DataFrame
        """
        if self.tmdb_api_key is None:
            print("TMDb API key not set. Skipping metadata enrichment.")
            return self.movies_df
        
        if self.movies_df is None:
            print("Please load and clean data first")
            return pd.DataFrame()
        
        print(f"Enriching metadata for up to {max_movies} movies...")
        
        enriched_movies = self.movies_df.copy()
        base_url = "https://api.themoviedb.org/3"
        
        for idx, row in tqdm(enriched_movies.head(max_movies).iterrows(), total=min(max_movies, len(enriched_movies))):
            try:
                # Search for movie by title and year
                search_url = f"{base_url}/search/movie"
                params = {
                    'api_key': self.tmdb_api_key,
                    'query': row['clean_title'],
                    'year': row['year'] if row['year'] > 0 else None
                }
                
                response = requests.get(search_url, params=params)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    
                    if results:
                        movie_id = results[0]['id']
                        
                        # Get detailed movie information
                        detail_url = f"{base_url}/movie/{movie_id}"
                        detail_params = {
                            'api_key': self.tmdb_api_key,
                            'append_to_response': 'credits,keywords'
                        }
                        
                        detail_response = requests.get(detail_url, params=detail_params)
                        if detail_response.status_code == 200:
                            movie_data = detail_response.json()
                            
                            # Extract plot
                            enriched_movies.at[idx, 'plot'] = movie_data.get('overview', '')
                            
                            # Extract cast (top 5 actors)
                            credits = movie_data.get('credits', {})
                            cast = credits.get('cast', [])
                            cast_names = [actor['name'] for actor in cast[:5]]
                            enriched_movies.at[idx, 'cast'] = '|'.join(cast_names)
                            
                            # Extract director
                            crew = credits.get('crew', [])
                            directors = [person['name'] for person in crew if person['job'] == 'Director']
                            enriched_movies.at[idx, 'director'] = '|'.join(directors[:3])
                            
                            # Extract keywords
                            keywords = movie_data.get('keywords', {}).get('keywords', [])
                            keyword_names = [kw['name'] for kw in keywords[:10]]
                            enriched_movies.at[idx, 'keywords'] = '|'.join(keyword_names)
                            
                            # Extract poster URL
                            poster_path = movie_data.get('poster_path')
                            if poster_path:
                                enriched_movies.at[idx, 'poster_url'] = f"https://image.tmdb.org/t/p/w500{poster_path}"
                
                # Rate limiting - be nice to the API
                import time
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error enriching movie {row['title']}: {e}")
                continue
        
        # Update combined features with enriched metadata
        enriched_movies['combined_features'] = (
            enriched_movies['genres'] + ' ' +
            enriched_movies['plot'] + ' ' +
            enriched_movies['cast'] + ' ' +
            enriched_movies['director'] + ' ' +
            enriched_movies['keywords']
        ).str.replace('|', ' ').fillna('')
        
        self.movies_df = enriched_movies
        print("Metadata enrichment completed!")
        return enriched_movies
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge movies and ratings datasets.
        
        Returns:
            Merged DataFrame
        """
        if self.movies_df is None or self.ratings_df is None:
            print("Please load data first using load_data()")
            return pd.DataFrame()
        
        print("Merging datasets...")
        
        # Merge ratings with movie information
        self.merged_df = self.ratings_df.merge(
            self.movies_df, 
            on='movieId', 
            how='left'
        )
        
        print(f"Merged dataset: {self.merged_df.shape[0]} ratings, {self.merged_df.shape[1]} features")
        return self.merged_df
    
    def save_processed_data(self, output_path: str = 'processed_data.csv'):
        """
        Save processed data to CSV file.
        
        Args:
            output_path: Path to save the processed data
        """
        if self.merged_df is not None:
            self.merged_df.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
        else:
            print("No processed data to save. Please run merge_datasets() first.")
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of the processed data.
        
        Returns:
            Dictionary with data summary
        """
        if self.merged_df is None:
            return {}
        
        summary = {
            'total_movies': self.movies_df['movieId'].nunique(),
            'total_users': self.ratings_df['userId'].nunique(),
            'total_ratings': len(self.ratings_df),
            'avg_rating': self.ratings_df['rating'].mean(),
            'rating_std': self.ratings_df['rating'].std(),
            'sparsity': 1 - (len(self.ratings_df) / (self.movies_df['movieId'].nunique() * self.ratings_df['userId'].nunique())),
            'movies_with_metadata': self.movies_df['plot'].str.len().gt(0).sum(),
            'movies_with_posters': self.movies_df['poster_url'].str.len().gt(0).sum()
        }
        
        return summary
    
    def get_popular_movies(self, min_ratings: int = 50) -> pd.DataFrame:
        """
        Get popular movies based on number of ratings.
        
        Args:
            min_ratings: Minimum number of ratings required
            
        Returns:
            DataFrame of popular movies
        """
        if self.merged_df is None:
            print("Please merge datasets first")
            return pd.DataFrame()
        
        movie_stats = self.merged_df.groupby('movieId').agg({
            'rating': ['count', 'mean'],
            'title': 'first',
            'genres': 'first'
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'rating_count', 'rating_mean', 'title', 'genres']
        popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings].sort_values('rating_count', ascending=False)
        
        return popular_movies

def main():
    """Example usage of the DataPreprocessor class."""
    preprocessor = DataPreprocessor()
    
    # Load data
    data = preprocessor.load_data()
    
    # Clean data
    cleaned_movies = preprocessor.clean_data()
    
    # Optional: Set TMDb API key for metadata enrichment
    # preprocessor.set_tmdb_api_key('your_api_key_here')
    # enriched_movies = preprocessor.enrich_metadata(max_movies=50)
    
    # Merge datasets
    merged_data = preprocessor.merge_datasets()
    
    # Get summary
    summary = preprocessor.get_data_summary()
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Get popular movies
    popular_movies = preprocessor.get_popular_movies(min_ratings=100)
    print(f"\nTop 10 popular movies:")
    print(popular_movies.head(10)[['title', 'rating_count', 'rating_mean']])

if __name__ == "__main__":
    main()
