"""
Main Module for Hybrid Movie Recommendation System

This module integrates all components:
- Data preprocessing
- Content-based filtering
- Collaborative filtering
- Hybrid recommendations
- User interfaces (Terminal and Streamlit)
"""
import sys
import os
import argparse
from typing import Optional

# Import our modules
from data_preprocessing import DataPreprocessor
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender
from ui import TerminalUI, StreamlitUI

def initialize_system(use_metadata_enrichment: bool = False, tmdb_api_key: Optional[str] = None):
    """
    Initialize the complete recommendation system.
    
    Args:
        use_metadata_enrichment: Whether to enrich metadata with TMDb API
        tmdb_api_key: TMDb API key for metadata enrichment
        
    Returns:
        Tuple of (hybrid_recommender, data_summary)
    """
    print("🎬 Initializing Hybrid Movie Recommendation System...")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    print("\n📊 Step 1: Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    
    if not data:
        print("❌ Failed to load data. Please check if movies.csv and ratings.csv exist.")
        return None, None
    
    cleaned_movies = preprocessor.clean_data()
    
    # Optional metadata enrichment
    if use_metadata_enrichment and tmdb_api_key:
        print("\n🔍 Step 1.5: Enriching metadata with TMDb API...")
        preprocessor.set_tmdb_api_key(tmdb_api_key)
        enriched_movies = preprocessor.enrich_metadata(max_movies=100)  # Limit for demo
        cleaned_movies = enriched_movies
    
    merged_data = preprocessor.merge_datasets()
    data_summary = preprocessor.get_data_summary()
    
    # Step 2: Initialize Content-Based Recommender
    print("\n🎯 Step 2: Initializing content-based recommender...")
    content_recommender = ContentBasedRecommender(cleaned_movies)
    content_recommender.fit()
    
    # Step 3: Initialize Collaborative Recommender
    print("\n👥 Step 3: Initializing collaborative recommender...")
    collab_recommender = CollaborativeRecommender(data['ratings'], cleaned_movies)
    collab_recommender.prepare_data()
    collab_recommender.fit_svd()
    
    # Step 4: Initialize Hybrid Recommender
    print("\n🔗 Step 4: Initializing hybrid recommender...")
    hybrid_recommender = HybridRecommender(content_recommender, collab_recommender)
    
    print("\n✅ System initialization completed!")
    print("=" * 60)
    
    return hybrid_recommender, data_summary

def run_terminal_interface(hybrid_recommender: HybridRecommender):
    """Run the terminal interface."""
    print("\n🚀 Starting Terminal Interface...")
    terminal_ui = TerminalUI(hybrid_recommender)
    terminal_ui.run()

def run_streamlit_interface(hybrid_recommender: HybridRecommender):
    """Run the Streamlit web interface."""
    print("\n🌐 Starting Streamlit Web Interface...")
    print("The web interface will open in your browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    
    # Create a simple Streamlit app
    import streamlit as st
    
    # Set page config
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize Streamlit UI
    streamlit_ui = StreamlitUI(hybrid_recommender)
    streamlit_ui.run()

def demo_mode(hybrid_recommender: HybridRecommender):
    """Run a quick demo of the system."""
    print("\n🎬 DEMO MODE - Quick System Test")
    print("=" * 50)
    
    # Test movie recommendations
    print("\n📊 Testing movie recommendations...")
    movie_title = "Toy Story"
    movie_recs = hybrid_recommender.get_hybrid_recommendations(
        movie_title=movie_title,
        num_recommendations=5,
        alpha=0.6
    )
    
    if movie_recs is not None:
        print(f"\nTop 5 recommendations for '{movie_title}':")
        for i, (_, row) in enumerate(movie_recs.iterrows(), 1):
            print(f"{i}. {row['title']} ({row.get('year', 'N/A')}) - {row['genres']}")
    
    # Test user recommendations
    print("\n👤 Testing user recommendations...")
    user_id = 1
    user_recs = hybrid_recommender.get_hybrid_recommendations(
        user_id=user_id,
        num_recommendations=5,
        alpha=0.4
    )
    
    if user_recs is not None:
        print(f"\nTop 5 recommendations for user {user_id}:")
        for i, (_, row) in enumerate(user_recs.iterrows(), 1):
            print(f"{i}. {row['title']} ({row.get('year', 'N/A')}) - {row['genres']}")
    
    # Test ensemble recommendations
    print("\n🎯 Testing ensemble recommendations...")
    ensemble_recs = hybrid_recommender.get_ensemble_recommendations(
        movie_title=movie_title,
        user_id=user_id,
        num_recommendations=5
    )
    
    if ensemble_recs is not None:
        print(f"\nTop 5 ensemble recommendations:")
        for i, (_, row) in enumerate(ensemble_recs.iterrows(), 1):
            print(f"{i}. {row['title']} ({row.get('year', 'N/A')}) - {row['genres']}")
    
    print("\n✅ Demo completed successfully!")

def main():
    """Main function to run the recommendation system."""
    parser = argparse.ArgumentParser(description="Hybrid Movie Recommendation System")
    parser.add_argument(
        "--interface", 
        choices=["terminal", "streamlit", "demo"], 
        default="terminal",
        help="Interface to use (default: terminal)"
    )
    parser.add_argument(
        "--enrich-metadata", 
        action="store_true",
        help="Enable metadata enrichment with TMDb API"
    )
    parser.add_argument(
        "--tmdb-api-key", 
        type=str,
        help="TMDb API key for metadata enrichment"
    )
    parser.add_argument(
        "--save-models", 
        action="store_true",
        help="Save trained models to disk"
    )
    parser.add_argument(
        "--load-models", 
        action="store_true",
        help="Load trained models from disk"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize the system
        hybrid_recommender, data_summary = initialize_system(
            use_metadata_enrichment=args.enrich_metadata,
            tmdb_api_key=args.tmdb_api_key
        )
        
        if hybrid_recommender is None:
            print("❌ Failed to initialize the system. Exiting.")
            return
        
        # Display system summary
        if data_summary:
            print("\n📊 SYSTEM SUMMARY:")
            for key, value in data_summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        # Save models if requested
        if args.save_models:
            print("\n💾 Saving models...")
            hybrid_recommender.content_recommender.save_model("models/content_model.pkl")
            hybrid_recommender.collaborative_recommender.save_model("models/collaborative_model.pkl")
            print("✅ Models saved successfully!")
        
        # Load models if requested
        if args.load_models:
            print("\n📂 Loading models...")
            if os.path.exists("models/content_model.pkl"):
                hybrid_recommender.content_recommender.load_model("models/content_model.pkl")
            if os.path.exists("models/collaborative_model.pkl"):
                hybrid_recommender.collaborative_recommender.load_model("models/collaborative_model.pkl")
            print("✅ Models loaded successfully!")
        
        # Run the appropriate interface
        if args.interface == "terminal":
            run_terminal_interface(hybrid_recommender)
        elif args.interface == "streamlit":
            run_streamlit_interface(hybrid_recommender)
        elif args.interface == "demo":
            demo_mode(hybrid_recommender)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()