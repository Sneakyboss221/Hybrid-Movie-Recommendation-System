"""
User Interface Module for Hybrid Movie Recommendation System

This module provides:
- Terminal/console interface with menu-driven options
- Streamlit web interface with interactive features
- Visualization capabilities
- User-friendly recommendation display
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, List
import os
import sys

# Import our modules
from data_preprocessing import DataPreprocessor
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender

class TerminalUI:
    """
    Terminal-based user interface for the recommendation system.
    """
    
    def __init__(self, hybrid_recommender: HybridRecommender):
        """
        Initialize the terminal UI.
        
        Args:
            hybrid_recommender: Fitted hybrid recommender
        """
        self.hybrid_recommender = hybrid_recommender
        self.content_recommender = hybrid_recommender.content_recommender
        self.collaborative_recommender = hybrid_recommender.collaborative_recommender
        
    def display_menu(self):
        """Display the main menu."""
        print("\n" + "="*60)
        print("🎬 HYBRID MOVIE RECOMMENDATION SYSTEM")
        print("="*60)
        print("1. Get movie recommendations by title")
        print("2. Get recommendations for a user")
        print("3. Compare different algorithms")
        print("4. View user profile")
        print("5. Get serendipitous recommendations")
        print("6. View system statistics")
        print("7. Exit")
        print("="*60)
    
    def get_movie_recommendations(self):
        """Get recommendations for a movie."""
        print("\n🎬 MOVIE RECOMMENDATIONS")
        print("-" * 40)
        
        # Get movie title from user
        movie_title = input("Enter a movie title: ").strip()
        if not movie_title:
            print("Please enter a valid movie title.")
            return
        
        # Get number of recommendations
        try:
            num_recs = int(input("Number of recommendations (default 10): ") or "10")
        except ValueError:
            num_recs = 10
        
        # Get alpha value for hybrid
        try:
            alpha = float(input("Content-based weight (0-1, default 0.5): ") or "0.5")
            alpha = max(0, min(1, alpha))  # Clamp between 0 and 1
        except ValueError:
            alpha = 0.5
        
        print(f"\nGetting recommendations for '{movie_title}'...")
        
        # Get content-based recommendations
        print("\n📊 CONTENT-BASED RECOMMENDATIONS:")
        content_recs = self.content_recommender.get_recommendations(movie_title, num_recs)
        if content_recs is not None:
            self._display_recommendations(content_recs, "Content-Based")
        
        # Get hybrid recommendations (content-based only since no user_id)
        print("\n🔗 HYBRID RECOMMENDATIONS (Content-Based Only):")
        hybrid_recs = self.hybrid_recommender.get_hybrid_recommendations(
            movie_title=movie_title, 
            num_recommendations=num_recs, 
            alpha=alpha
        )
        if hybrid_recs is not None:
            self._display_recommendations(hybrid_recs, "Hybrid")
            

        
        # Show diversity score
        if hybrid_recs is not None:
            diversity = self.hybrid_recommender.get_recommendation_diversity(hybrid_recs)
            print(f"\n🎯 RECOMMENDATION DIVERSITY: {diversity:.3f}")
    
    def get_user_recommendations(self):
        """Get recommendations for a user."""
        print("\n👤 USER RECOMMENDATIONS")
        print("-" * 40)
        
        # Get user ID from user
        try:
            user_id = int(input("Enter user ID: "))
        except ValueError:
            print("Please enter a valid user ID.")
            return
        
        # Get number of recommendations
        try:
            num_recs = int(input("Number of recommendations (default 10): ") or "10")
        except ValueError:
            num_recs = 10
        
        # Get alpha value for hybrid
        try:
            alpha = float(input("Content-based weight (0-1, default 0.5): ") or "0.5")
            alpha = max(0, min(1, alpha))
        except ValueError:
            alpha = 0.5
        
        print(f"\nGetting recommendations for user {user_id}...")
        
        # Get collaborative recommendations
        print("\n👥 COLLABORATIVE RECOMMENDATIONS:")
        collab_recs = self.collaborative_recommender.get_user_recommendations(user_id, num_recs)
        if collab_recs is not None:
            self._display_recommendations(collab_recs, "Collaborative")
        
        # Get hybrid recommendations (collaborative only since no movie_title)
        print("\n🔗 HYBRID RECOMMENDATIONS (Collaborative Only):")
        hybrid_recs = self.hybrid_recommender.get_hybrid_recommendations(
            user_id=user_id, 
            num_recommendations=num_recs, 
            alpha=alpha
        )
        if hybrid_recs is not None:
            self._display_recommendations(hybrid_recs, "Hybrid")
    
    def compare_algorithms(self):
        """Compare different recommendation algorithms."""
        print("\n⚖️ ALGORITHM COMPARISON")
        print("-" * 40)
        
        # Get input parameters
        movie_title = input("Enter a movie title (optional): ").strip()
        user_id_input = input("Enter a user ID (optional): ").strip()
        
        try:
            user_id = int(user_id_input) if user_id_input else None
        except ValueError:
            user_id = None
        
        if not movie_title and not user_id:
            print("Please provide either a movie title or user ID.")
            return
        
        num_recs = 5
        
        results = {}
        
        # Content-based (if movie title provided)
        if movie_title:
            print(f"\n📊 CONTENT-BASED RECOMMENDATIONS:")
            content_recs = self.content_recommender.get_recommendations(movie_title, num_recs)
            if content_recs is not None:
                results['Content-Based'] = content_recs
                self._display_recommendations(content_recs, "Content-Based")
        
        # Collaborative (if user ID provided)
        if user_id:
            print(f"\n👥 COLLABORATIVE RECOMMENDATIONS:")
            collab_recs = self.collaborative_recommender.get_user_recommendations(user_id, num_recs)
            if collab_recs is not None:
                results['Collaborative'] = collab_recs
                self._display_recommendations(collab_recs, "Collaborative")
        
        # Hybrid (if both provided)
        if movie_title and user_id:
            print(f"\n🔗 HYBRID RECOMMENDATIONS:")
            hybrid_recs = self.hybrid_recommender.get_hybrid_recommendations(
                movie_title=movie_title,
                user_id=user_id,
                num_recommendations=num_recs,
                alpha=0.5
            )
            if hybrid_recs is not None:
                results['Hybrid'] = hybrid_recs
                self._display_recommendations(hybrid_recs, "Hybrid")
        
        # Ensemble
        print(f"\n🎯 ENSEMBLE RECOMMENDATIONS:")
        ensemble_recs = self.hybrid_recommender.get_ensemble_recommendations(
            movie_title=movie_title,
            user_id=user_id,
            num_recommendations=num_recs
        )
        if ensemble_recs is not None:
            results['Ensemble'] = ensemble_recs
            self._display_recommendations(ensemble_recs, "Ensemble")
        
        # Compare diversity scores
        print("\n🎯 DIVERSITY COMPARISON:")
        for algo, recs in results.items():
            diversity = self.hybrid_recommender.get_recommendation_diversity(recs)
            print(f"{algo}: {diversity:.3f}")
    
    def view_user_profile(self):
        """View user profile information."""
        print("\n👤 USER PROFILE")
        print("-" * 40)
        
        try:
            user_id = int(input("Enter user ID: "))
        except ValueError:
            print("Please enter a valid user ID.")
            return
        
        profile = self.collaborative_recommender.get_user_profile(user_id)
        
        if not profile:
            print(f"User {user_id} not found or has no ratings.")
            return
        
        print(f"\n📊 USER {user_id} PROFILE:")
        print(f"Total ratings: {profile['total_ratings']}")
        print(f"Average rating: {profile['avg_rating']:.2f}")
        print(f"Rating std dev: {profile['rating_std']:.2f}")
        
        print(f"\n🎭 FAVORITE GENRES:")
        for genre, avg_rating in profile['favorite_genres']:
            print(f"  {genre}: {avg_rating:.2f}")
        
        print(f"\n📈 RATING DISTRIBUTION:")
        for rating, count in profile['rating_distribution'].items():
            print(f"  {rating} stars: {count} movies")
    
    def get_serendipitous_recommendations(self):
        """Get serendipitous recommendations."""
        print("\n🎲 SERENDIPITOUS RECOMMENDATIONS")
        print("-" * 40)
        
        movie_title = input("Enter a movie title: ").strip()
        if not movie_title:
            print("Please enter a valid movie title.")
            return
        
        try:
            user_id = int(input("Enter user ID: "))
        except ValueError:
            print("Please enter a valid user ID.")
            return
        
        try:
            serendipity_weight = float(input("Serendipity weight (0-1, default 0.3): ") or "0.3")
            serendipity_weight = max(0, min(1, serendipity_weight))
        except ValueError:
            serendipity_weight = 0.3
        
        num_recs = 10
        
        print(f"\nGetting serendipitous recommendations...")
        
        serendipitous_recs = self.hybrid_recommender.get_serendipitous_recommendations(
            movie_title=movie_title,
            user_id=user_id,
            num_recommendations=num_recs,
            serendipity_weight=serendipity_weight
        )
        
        if serendipitous_recs is not None:
            self._display_recommendations(serendipitous_recs, "Serendipitous")
            
            # Show serendipity scores
            print(f"\n🎲 SERENDIPITY SCORES:")
            for _, row in serendipitous_recs.iterrows():
                print(f"  {row['title']}: {row['serendipity_score']:.3f}")
    
    def view_system_statistics(self):
        """View system statistics."""
        print("\n📊 SYSTEM STATISTICS")
        print("-" * 40)
        
        # Data statistics
        print(f"Total movies: {len(self.content_recommender.movies_df)}")
        print(f"Total users: {self.collaborative_recommender.ratings_df['userId'].nunique()}")
        print(f"Total ratings: {len(self.collaborative_recommender.ratings_df)}")
        print(f"Average rating: {self.collaborative_recommender.ratings_df['rating'].mean():.2f}")
        
        # Sparsity
        sparsity = 1 - (len(self.collaborative_recommender.ratings_df) / 
                        (self.collaborative_recommender.ratings_df['userId'].nunique() * 
                         self.content_recommender.movies_df['movieId'].nunique()))
        print(f"Data sparsity: {sparsity:.3f}")
        
        # Genre distribution
        all_genres = []
        for genres in self.content_recommender.movies_df['genres']:
            all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts()
        print(f"\n🎭 TOP 10 GENRES:")
        for genre, count in genre_counts.head(10).items():
            print(f"  {genre}: {count} movies")
    
    def _display_recommendations(self, recommendations: pd.DataFrame, algo_name: str):
        """Display recommendations in a formatted way."""
        if recommendations is None or len(recommendations) == 0:
            print("No recommendations found.")
            return
        
        print(f"\n{algo_name.upper()} RECOMMENDATIONS:")
        print("-" * 60)
        
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            title = row['title']
            genres = row['genres']
            year = row.get('year', 'N/A')
            
            # Get score based on algorithm
            if 'hybrid_score' in row:
                score = row['hybrid_score']
                score_type = "Hybrid Score"
            elif 'content_score' in row:
                score = row['content_score']
                score_type = "Similarity"
            elif 'collaborative_score' in row:
                score = row['collaborative_score']
                score_type = "Predicted Rating"
            elif 'ensemble_score' in row:
                score = row['ensemble_score']
                score_type = "Ensemble Score"
            else:
                score = "N/A"
                score_type = "Score"
            
            print(f"{i:2d}. {title} ({year})")
            print(f"    Genres: {genres}")
            if score != "N/A":
                print(f"    {score_type}: {score:.3f}")
            print()
    
    def run(self):
        """Run the terminal interface."""
        while True:
            self.display_menu()
            
            try:
                choice = input("Enter your choice (1-7): ").strip()
                
                if choice == '1':
                    self.get_movie_recommendations()
                elif choice == '2':
                    self.get_user_recommendations()
                elif choice == '3':
                    self.compare_algorithms()
                elif choice == '4':
                    self.view_user_profile()
                elif choice == '5':
                    self.get_serendipitous_recommendations()
                elif choice == '6':
                    self.view_system_statistics()
                elif choice == '7':
                    print("\n👋 Thank you for using the Movie Recommendation System!")
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 7.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                input("Press Enter to continue...")

class StreamlitUI:
    """
    Streamlit web interface for the recommendation system.
    """
    
    def __init__(self, hybrid_recommender: HybridRecommender):
        """
        Initialize the Streamlit UI.
        
        Args:
            hybrid_recommender: Fitted hybrid recommender
        """
        self.hybrid_recommender = hybrid_recommender
        self.content_recommender = hybrid_recommender.content_recommender
        self.collaborative_recommender = hybrid_recommender.collaborative_recommender
        
        # Set page config
        st.set_page_config(
            page_title="Movie Recommendation System",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the Streamlit interface."""
        st.title("🎬 Hybrid Movie Recommendation System")
        st.markdown("---")
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Movie Recommendations", 
            "👤 User Recommendations", 
            "⚖️ Algorithm Comparison",
            "📊 Analytics",
            "🎲 Serendipitous"
        ])
        
        with tab1:
            self._movie_recommendations_tab()
        
        with tab2:
            self._user_recommendations_tab()
        
        with tab3:
            self._algorithm_comparison_tab()
        
        with tab4:
            self._analytics_tab()
        
        with tab5:
            self._serendipitous_tab()
    
    def _create_sidebar(self):
        """Create the sidebar with controls."""
        st.sidebar.header("⚙️ Settings")
        
        # Alpha slider
        st.sidebar.subheader("Hybrid Weight")
        alpha = st.sidebar.slider(
            "Content-based weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Weight for content-based filtering in hybrid recommendations"
        )
        
        # Number of recommendations
        num_recs = st.sidebar.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        # Store in session state
        st.session_state.alpha = alpha
        st.session_state.num_recs = num_recs
        
        # System info
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 System Info")
        st.sidebar.metric("Movies", len(self.content_recommender.movies_df))
        st.sidebar.metric("Users", self.collaborative_recommender.ratings_df['userId'].nunique())
        st.sidebar.metric("Ratings", len(self.collaborative_recommender.ratings_df))
    
    def _movie_recommendations_tab(self):
        """Movie recommendations tab."""
        st.header("🎯 Movie Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Movie title input with autocomplete
            movie_titles = self.content_recommender.movies_df['clean_title'].tolist()
            movie_title = st.selectbox(
                "Select a movie:",
                options=movie_titles,
                index=movie_titles.index("Toy Story") if "Toy Story" in movie_titles else 0
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("Get Recommendations", type="primary"):
                st.session_state.movie_recs = self._get_movie_recommendations(movie_title)
        
        # Display recommendations
        if hasattr(st.session_state, 'movie_recs') and st.session_state.movie_recs is not None:
            self._display_recommendations_web(st.session_state.movie_recs, "Movie Recommendations")
    
    def _user_recommendations_tab(self):
        """User recommendations tab."""
        st.header("👤 User Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_id = st.number_input(
                "Enter user ID:",
                min_value=1,
                max_value=self.collaborative_recommender.ratings_df['userId'].max(),
                value=1,
                step=1
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("Get Recommendations", type="primary"):
                st.session_state.user_recs = self._get_user_recommendations(user_id)
                st.session_state.user_profile = self.collaborative_recommender.get_user_profile(user_id)
        
        # Display recommendations
        if hasattr(st.session_state, 'user_recs') and st.session_state.user_recs is not None:
            self._display_recommendations_web(st.session_state.user_recs, "User Recommendations")
        
        # Display user profile
        if hasattr(st.session_state, 'user_profile') and st.session_state.user_profile:
            self._display_user_profile_web(st.session_state.user_profile)
    
    def _algorithm_comparison_tab(self):
        """Algorithm comparison tab."""
        st.header("⚖️ Algorithm Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            movie_titles = self.content_recommender.movies_df['clean_title'].tolist()
            movie_title = st.selectbox(
                "Select a movie (optional):",
                options=[""] + movie_titles,
                index=0
            )
        
        with col2:
            user_id = st.number_input(
                "Enter user ID (optional):",
                min_value=1,
                max_value=self.collaborative_recommender.ratings_df['userId'].max(),
                value=1,
                step=1
            )
        
        if st.button("Compare Algorithms", type="primary"):
            results = self._compare_algorithms(movie_title if movie_title else None, user_id)
            if results:
                self._display_algorithm_comparison(results)
    
    def _analytics_tab(self):
        """Analytics tab."""
        st.header("📊 Analytics")
        
        # System statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movies", len(self.content_recommender.movies_df))
        
        with col2:
            st.metric("Total Users", self.collaborative_recommender.ratings_df['userId'].nunique())
        
        with col3:
            st.metric("Total Ratings", len(self.collaborative_recommender.ratings_df))
        
        with col4:
            avg_rating = self.collaborative_recommender.ratings_df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}")
        
        # Genre distribution
        st.subheader("🎭 Genre Distribution")
        all_genres = []
        for genres in self.content_recommender.movies_df['genres']:
            all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts().head(15)
        
        fig = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation='h',
            title="Top 15 Movie Genres"
        )
        fig.update_layout(xaxis_title="Number of Movies", yaxis_title="Genre")
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating distribution
        st.subheader("📈 Rating Distribution")
        rating_counts = self.collaborative_recommender.ratings_df['rating'].value_counts().sort_index()
        
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Rating Distribution"
        )
        fig.update_layout(xaxis_title="Rating", yaxis_title="Number of Ratings")
        st.plotly_chart(fig, use_container_width=True)
    
    def _serendipitous_tab(self):
        """Serendipitous recommendations tab."""
        st.header("🎲 Serendipitous Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            movie_titles = self.content_recommender.movies_df['clean_title'].tolist()
            movie_title = st.selectbox(
                "Select a movie:",
                options=movie_titles,
                index=movie_titles.index("Toy Story") if "Toy Story" in movie_titles else 0
            )
        
        with col2:
            user_id = st.number_input(
                "Enter user ID:",
                min_value=1,
                max_value=self.collaborative_recommender.ratings_df['userId'].max(),
                value=1,
                step=1
            )
        
        with col3:
            serendipity_weight = st.slider(
                "Serendipity weight:",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
        
        if st.button("Get Serendipitous Recommendations", type="primary"):
            serendipitous_recs = self.hybrid_recommender.get_serendipitous_recommendations(
                movie_title=movie_title,
                user_id=user_id,
                num_recommendations=st.session_state.num_recs,
                serendipity_weight=serendipity_weight
            )
            
            if serendipitous_recs is not None:
                self._display_recommendations_web(serendipitous_recs, "Serendipitous Recommendations")
    
    def _get_movie_recommendations(self, movie_title: str):
        """Get movie recommendations."""
        with st.spinner("Getting recommendations..."):
            return self.hybrid_recommender.get_hybrid_recommendations(
                movie_title=movie_title,
                num_recommendations=st.session_state.num_recs,
                alpha=st.session_state.alpha
            )
    
    def _get_user_recommendations(self, user_id: int):
        """Get user recommendations."""
        with st.spinner("Getting recommendations..."):
            return self.hybrid_recommender.get_hybrid_recommendations(
                user_id=user_id,
                num_recommendations=st.session_state.num_recs,
                alpha=st.session_state.alpha
            )
    
    def _compare_algorithms(self, movie_title: str = None, user_id: int = None):
        """Compare different algorithms."""
        if not movie_title and not user_id:
            st.error("Please provide either a movie title or user ID.")
            return None
        
        with st.spinner("Comparing algorithms..."):
            results = {}
            
            if movie_title:
                content_recs = self.content_recommender.get_recommendations(movie_title, 5)
                if content_recs is not None:
                    results['Content-Based'] = content_recs
            
            if user_id:
                collab_recs = self.collaborative_recommender.get_user_recommendations(user_id, 5)
                if collab_recs is not None:
                    results['Collaborative'] = collab_recs
            
            if movie_title and user_id:
                hybrid_recs = self.hybrid_recommender.get_hybrid_recommendations(
                    movie_title=movie_title,
                    user_id=user_id,
                    num_recommendations=5,
                    alpha=st.session_state.alpha
                )
                if hybrid_recs is not None:
                    results['Hybrid'] = hybrid_recs
            
            ensemble_recs = self.hybrid_recommender.get_ensemble_recommendations(
                movie_title=movie_title,
                user_id=user_id,
                num_recommendations=5
            )
            if ensemble_recs is not None:
                results['Ensemble'] = ensemble_recs
            
            return results
    
    def _display_recommendations_web(self, recommendations: pd.DataFrame, title: str):
        """Display recommendations in web format."""
        if recommendations is None or len(recommendations) == 0:
            st.warning("No recommendations found.")
            return
        
        st.subheader(title)
        
        # Create a nice display
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{i}. {row['title']}** ({row.get('year', 'N/A')})")
                    st.write(f"*{row['genres']}*")
                
                with col2:
                    if 'hybrid_score' in row:
                        st.metric("Score", f"{row['hybrid_score']:.3f}")
                    elif 'content_score' in row:
                        st.metric("Similarity", f"{row['content_score']:.3f}")
                    elif 'collaborative_score' in row:
                        st.metric("Predicted", f"{row['collaborative_score']:.3f}")
                
                st.divider()
    
    def _display_user_profile_web(self, profile: Dict):
        """Display user profile in web format."""
        st.subheader("👤 User Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Ratings", profile['total_ratings'])
            st.metric("Average Rating", f"{profile['avg_rating']:.2f}")
            st.metric("Rating Std Dev", f"{profile['rating_std']:.2f}")
        
        with col2:
            st.write("**Favorite Genres:**")
            for genre, avg_rating in profile['favorite_genres']:
                st.write(f"• {genre}: {avg_rating:.2f}")
        
        # Rating distribution chart
        rating_dist = profile['rating_distribution']
        fig = px.bar(
            x=list(rating_dist.keys()),
            y=list(rating_dist.values()),
            title="Rating Distribution"
        )
        fig.update_layout(xaxis_title="Rating", yaxis_title="Number of Movies")
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_algorithm_comparison(self, results: Dict[str, pd.DataFrame]):
        """Display algorithm comparison results."""
        st.subheader("Algorithm Comparison Results")
        
        # Create tabs for each algorithm
        tabs = st.tabs(list(results.keys()))
        
        for tab, (algo_name, recs) in zip(tabs, results.items()):
            with tab:
                self._display_recommendations_web(recs, f"{algo_name} Recommendations")
        
        # Compare diversity scores
        st.subheader("Diversity Comparison")
        diversity_scores = {}
        for algo_name, recs in results.items():
            diversity = self.hybrid_recommender.get_recommendation_diversity(recs)
            diversity_scores[algo_name] = diversity
        
        fig = px.bar(
            x=list(diversity_scores.keys()),
            y=list(diversity_scores.values()),
            title="Recommendation Diversity by Algorithm"
        )
        fig.update_layout(xaxis_title="Algorithm", yaxis_title="Diversity Score")
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Example usage of the UI classes."""
    from data_preprocessing import DataPreprocessor
    from content_based import ContentBasedRecommender
    from collaborative import CollaborativeRecommender
    from hybrid import HybridRecommender
    
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
    
    # Run terminal UI
    print("Starting Terminal UI...")
    terminal_ui = TerminalUI(hybrid_recommender)
    terminal_ui.run()

if __name__ == "__main__":
    main()
