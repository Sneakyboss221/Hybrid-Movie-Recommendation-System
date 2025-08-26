"""
Streamlit Web Application for Hybrid Movie Recommendation System

This file provides a standalone Streamlit web interface for the recommendation system.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Import our modules
from data_preprocessing import DataPreprocessor
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender

# Set page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_system():
    """Initialize the recommendation system (cached for performance)."""
    with st.spinner("Initializing recommendation system..."):
        # Load and preprocess data
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data()
        
        if not data:
            st.error("Failed to load data. Please check if movies.csv and ratings.csv exist.")
            return None, None
        
        cleaned_movies = preprocessor.clean_data()
        merged_data = preprocessor.merge_datasets()
        data_summary = preprocessor.get_data_summary()
        
        # Initialize recommenders
        content_recommender = ContentBasedRecommender(cleaned_movies)
        content_recommender.fit()
        
        collab_recommender = CollaborativeRecommender(data['ratings'], cleaned_movies)
        collab_recommender.prepare_data()
        collab_recommender.fit_svd()
        
        hybrid_recommender = HybridRecommender(content_recommender, collab_recommender)
        
        return hybrid_recommender, data_summary

def main():
    """Main Streamlit application."""
    st.title("🎬 Hybrid Movie Recommendation System")
    st.markdown("---")
    
    # Initialize system
    hybrid_recommender, data_summary = initialize_system()
    
    if hybrid_recommender is None:
        st.error("Failed to initialize the system. Please check your data files.")
        return
    
    # Sidebar
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
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Info")
    if data_summary:
        st.sidebar.metric("Movies", data_summary['total_movies'])
        st.sidebar.metric("Users", data_summary['total_users'])
        st.sidebar.metric("Ratings", data_summary['total_ratings'])
        st.sidebar.metric("Avg Rating", f"{data_summary['avg_rating']:.2f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Movie Recommendations", 
        "👤 User Recommendations", 
        "⚖️ Algorithm Comparison",
        "📊 Analytics",
        "🎲 Serendipitous"
    ])
    
    with tab1:
        movie_recommendations_tab(hybrid_recommender, alpha, num_recs)
    
    with tab2:
        user_recommendations_tab(hybrid_recommender, alpha, num_recs)
    
    with tab3:
        algorithm_comparison_tab(hybrid_recommender, alpha, num_recs)
    
    with tab4:
        analytics_tab(hybrid_recommender, data_summary)
    
    with tab5:
        serendipitous_tab(hybrid_recommender, num_recs)

def movie_recommendations_tab(hybrid_recommender, alpha, num_recs):
    """Movie recommendations tab."""
    st.header("🎯 Movie Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Movie title input with autocomplete
        movie_titles = hybrid_recommender.content_recommender.movies_df['clean_title'].tolist()
        movie_title = st.selectbox(
            "Select a movie:",
            options=movie_titles,
            index=movie_titles.index("Toy Story") if "Toy Story" in movie_titles else 0
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Getting recommendations..."):
                recommendations = hybrid_recommender.get_hybrid_recommendations(
                    movie_title=movie_title,
                    num_recommendations=num_recs,
                    alpha=alpha
                )
                
                if recommendations is not None:
                    st.session_state.movie_recs = recommendations
                    st.session_state.movie_title = movie_title
    
    # Display recommendations
    if hasattr(st.session_state, 'movie_recs') and st.session_state.movie_recs is not None:
        display_recommendations(st.session_state.movie_recs, f"Recommendations for '{st.session_state.movie_title}'")
        
        # Show explanation for top recommendation
        if len(st.session_state.movie_recs) > 0:
            explanation = hybrid_recommender.explain_recommendation(
                st.session_state.movie_title,
                st.session_state.movie_recs.iloc[0]['title']
            )
            st.info(f"💡 **Explanation:** {explanation}")
        
        # Show diversity score
        diversity = hybrid_recommender.get_recommendation_diversity(st.session_state.movie_recs)
        st.metric("Recommendation Diversity", f"{diversity:.3f}")

def user_recommendations_tab(hybrid_recommender, alpha, num_recs):
    """User recommendations tab."""
    st.header("👤 User Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.number_input(
            "Enter user ID:",
            min_value=1,
            max_value=hybrid_recommender.collaborative_recommender.ratings_df['userId'].max(),
            value=1,
            step=1
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Getting recommendations..."):
                recommendations = hybrid_recommender.get_hybrid_recommendations(
                    user_id=user_id,
                    num_recommendations=num_recs,
                    alpha=alpha
                )
                
                if recommendations is not None:
                    st.session_state.user_recs = recommendations
                    st.session_state.user_id = user_id
                
                # Get user profile
                user_profile = hybrid_recommender.collaborative_recommender.get_user_profile(user_id)
                if user_profile:
                    st.session_state.user_profile = user_profile
    
    # Display recommendations
    if hasattr(st.session_state, 'user_recs') and st.session_state.user_recs is not None:
        display_recommendations(st.session_state.user_recs, f"Recommendations for User {st.session_state.user_id}")
    
    # Display user profile
    if hasattr(st.session_state, 'user_profile') and st.session_state.user_profile:
        display_user_profile(st.session_state.user_profile)

def algorithm_comparison_tab(hybrid_recommender, alpha, num_recs):
    """Algorithm comparison tab."""
    st.header("⚖️ Algorithm Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        movie_titles = hybrid_recommender.content_recommender.movies_df['clean_title'].tolist()
        movie_title = st.selectbox(
            "Select a movie (optional):",
            options=[""] + movie_titles,
            index=0
        )
    
    with col2:
        user_id = st.number_input(
            "Enter user ID (optional):",
            min_value=1,
            max_value=hybrid_recommender.collaborative_recommender.ratings_df['userId'].max(),
            value=1,
            step=1
        )
    
    if st.button("Compare Algorithms", type="primary"):
        if not movie_title and not user_id:
            st.error("Please provide either a movie title or user ID.")
            return
        
        with st.spinner("Comparing algorithms..."):
            results = compare_algorithms(hybrid_recommender, movie_title if movie_title else None, user_id, alpha, num_recs)
            if results:
                display_algorithm_comparison(results, hybrid_recommender)

def analytics_tab(hybrid_recommender, data_summary):
    """Analytics tab."""
    st.header("📊 Analytics")
    
    if not data_summary:
        st.warning("No data summary available.")
        return
    
    # System statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Movies", data_summary['total_movies'])
    
    with col2:
        st.metric("Total Users", data_summary['total_users'])
    
    with col3:
        st.metric("Total Ratings", data_summary['total_ratings'])
    
    with col4:
        st.metric("Average Rating", f"{data_summary['avg_rating']:.2f}")
    
    # Genre distribution
    st.subheader("🎭 Genre Distribution")
    all_genres = []
    for genres in hybrid_recommender.content_recommender.movies_df['genres']:
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
    rating_counts = hybrid_recommender.collaborative_recommender.ratings_df['rating'].value_counts().sort_index()
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        title="Rating Distribution"
    )
    fig.update_layout(xaxis_title="Rating", yaxis_title="Number of Ratings")
    st.plotly_chart(fig, use_container_width=True)
    
    # Data sparsity
    st.subheader("📊 Data Sparsity")
    sparsity = data_summary['sparsity']
    st.metric("Data Sparsity", f"{sparsity:.3f}")
    st.info(f"This means {sparsity:.1%} of possible user-movie ratings are missing.")

def serendipitous_tab(hybrid_recommender, num_recs):
    """Serendipitous recommendations tab."""
    st.header("🎲 Serendipitous Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        movie_titles = hybrid_recommender.content_recommender.movies_df['clean_title'].tolist()
        movie_title = st.selectbox(
            "Select a movie:",
            options=movie_titles,
            index=movie_titles.index("Toy Story") if "Toy Story" in movie_titles else 0
        )
    
    with col2:
        user_id = st.number_input(
            "Enter user ID:",
            min_value=1,
            max_value=hybrid_recommender.collaborative_recommender.ratings_df['userId'].max(),
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
        with st.spinner("Getting serendipitous recommendations..."):
            serendipitous_recs = hybrid_recommender.get_serendipitous_recommendations(
                movie_title=movie_title,
                user_id=user_id,
                num_recommendations=num_recs,
                serendipity_weight=serendipity_weight
            )
            
            if serendipitous_recs is not None:
                display_recommendations(serendipitous_recs, "Serendipitous Recommendations")
                
                # Show serendipity scores
                st.subheader("🎲 Serendipity Scores")
                for _, row in serendipitous_recs.iterrows():
                    st.write(f"**{row['title']}**: {row['serendipity_score']:.3f}")

def display_recommendations(recommendations, title):
    """Display recommendations in a nice format."""
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
                elif 'ensemble_score' in row:
                    st.metric("Ensemble", f"{row['ensemble_score']:.3f}")
            
            st.divider()

def display_user_profile(profile):
    """Display user profile information."""
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

def compare_algorithms(hybrid_recommender, movie_title, user_id, alpha, num_recs):
    """Compare different recommendation algorithms."""
    results = {}
    
    if movie_title:
        content_recs = hybrid_recommender.content_recommender.get_recommendations(movie_title, num_recs)
        if content_recs is not None:
            results['Content-Based'] = content_recs
    
    if user_id:
        collab_recs = hybrid_recommender.collaborative_recommender.get_user_recommendations(user_id, num_recs)
        if collab_recs is not None:
            results['Collaborative'] = collab_recs
    
    if movie_title and user_id:
        hybrid_recs = hybrid_recommender.get_hybrid_recommendations(
            movie_title=movie_title,
            user_id=user_id,
            num_recommendations=num_recs,
            alpha=alpha
        )
        if hybrid_recs is not None:
            results['Hybrid'] = hybrid_recs
    
    ensemble_recs = hybrid_recommender.get_ensemble_recommendations(
        movie_title=movie_title,
        user_id=user_id,
        num_recommendations=num_recs
    )
    if ensemble_recs is not None:
        results['Ensemble'] = ensemble_recs
    
    return results

def display_algorithm_comparison(results, hybrid_recommender):
    """Display algorithm comparison results."""
    st.subheader("Algorithm Comparison Results")
    
    # Create tabs for each algorithm
    tabs = st.tabs(list(results.keys()))
    
    for tab, (algo_name, recs) in zip(tabs, results.items()):
        with tab:
            display_recommendations(recs, f"{algo_name} Recommendations")
    
    # Compare diversity scores
    st.subheader("Diversity Comparison")
    diversity_scores = {}
    for algo_name, recs in results.items():
        diversity = hybrid_recommender.get_recommendation_diversity(recs)
        diversity_scores[algo_name] = diversity
    
    fig = px.bar(
        x=list(diversity_scores.keys()),
        y=list(diversity_scores.values()),
        title="Recommendation Diversity by Algorithm"
    )
    fig.update_layout(xaxis_title="Algorithm", yaxis_title="Diversity Score")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()