"""
Test Script for Hybrid Movie Recommendation System

This script tests all components of the system to ensure they work correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

def test_data_preprocessing():
    """Test data preprocessing module."""
    print("🧪 Testing Data Preprocessing...")
    
    try:
        from data_preprocessing import DataPreprocessor
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load data
        data = preprocessor.load_data()
        assert data, "Failed to load data"
        assert 'movies' in data, "Movies data not found"
        assert 'ratings' in data, "Ratings data not found"
        
        # Clean data
        cleaned_movies = preprocessor.clean_data()
        assert len(cleaned_movies) > 0, "No movies after cleaning"
        assert 'clean_title' in cleaned_movies.columns, "clean_title column missing"
        assert 'combined_features' in cleaned_movies.columns, "combined_features column missing"
        
        # Merge datasets
        merged_data = preprocessor.merge_datasets()
        assert len(merged_data) > 0, "No data after merging"
        
        # Get summary
        summary = preprocessor.get_data_summary()
        assert summary, "No summary generated"
        
        print("✅ Data preprocessing tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        return False

def test_content_based():
    """Test content-based filtering module."""
    print("🧪 Testing Content-Based Filtering...")
    
    try:
        from data_preprocessing import DataPreprocessor
        from content_based import ContentBasedRecommender
        
        # Load and preprocess data
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data()
        cleaned_movies = preprocessor.clean_data()
        
        # Initialize content-based recommender
        content_recommender = ContentBasedRecommender(cleaned_movies)
        content_recommender.fit()
        
        # Test recommendations
        recommendations = content_recommender.get_recommendations("Toy Story", num_recommendations=5)
        assert recommendations is not None, "No recommendations generated"
        assert len(recommendations) > 0, "Empty recommendations"
        
        # Test explanation
        if len(recommendations) > 0:
            explanation = content_recommender.explain_recommendation("Toy Story", recommendations.iloc[0]['title'])
            assert explanation is not None, "No explanation generated"
        
        print("✅ Content-based filtering tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Content-based filtering test failed: {e}")
        return False

def test_collaborative():
    """Test collaborative filtering module."""
    print("🧪 Testing Collaborative Filtering...")
    
    try:
        from data_preprocessing import DataPreprocessor
        from collaborative import CollaborativeRecommender
        
        # Load and preprocess data
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data()
        cleaned_movies = preprocessor.clean_data()
        
        # Initialize collaborative recommender
        collab_recommender = CollaborativeRecommender(data['ratings'], cleaned_movies)
        collab_recommender.prepare_data()
        collab_recommender.fit_svd()
        
        # Test user recommendations
        user_recommendations = collab_recommender.get_user_recommendations(user_id=1, num_recommendations=5)
        # Note: This might be None if user has rated all movies, which is fine
        
        # Test user profile
        profile = collab_recommender.get_user_profile(user_id=1)
        assert profile, "No user profile generated"
        assert 'total_ratings' in profile, "Profile missing total_ratings"
        
        # Test evaluation
        metrics = collab_recommender.evaluate_model('svd')
        assert metrics, "No evaluation metrics"
        assert 'rmse' in metrics, "RMSE metric missing"
        
        print("✅ Collaborative filtering tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Collaborative filtering test failed: {e}")
        return False

def test_hybrid():
    """Test hybrid recommendation module."""
    print("🧪 Testing Hybrid Recommendations...")
    
    try:
        from data_preprocessing import DataPreprocessor
        from content_based import ContentBasedRecommender
        from collaborative import CollaborativeRecommender
        from hybrid import HybridRecommender
        
        # Load and preprocess data
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data()
        cleaned_movies = preprocessor.clean_data()
        
        # Initialize recommenders
        content_recommender = ContentBasedRecommender(cleaned_movies)
        content_recommender.fit()
        
        collab_recommender = CollaborativeRecommender(data['ratings'], cleaned_movies)
        collab_recommender.prepare_data()
        collab_recommender.fit_svd()
        
        # Initialize hybrid recommender
        hybrid_recommender = HybridRecommender(content_recommender, collab_recommender)
        
        # Test hybrid recommendations
        hybrid_recs = hybrid_recommender.get_hybrid_recommendations(
            movie_title="Toy Story",
            num_recommendations=5,
            alpha=0.6
        )
        assert hybrid_recs is not None, "No hybrid recommendations"
        assert len(hybrid_recs) > 0, "Empty hybrid recommendations"
        
        # Test ensemble recommendations
        ensemble_recs = hybrid_recommender.get_ensemble_recommendations(
            movie_title="Toy Story",
            num_recommendations=5
        )
        assert ensemble_recs is not None, "No ensemble recommendations"
        
        # Test explanation
        if len(hybrid_recs) > 0:
            explanation = hybrid_recommender.explain_recommendation(
                "Toy Story", 
                hybrid_recs.iloc[0]['title']
            )
            assert explanation is not None, "No explanation generated"
        
        # Test diversity
        diversity = hybrid_recommender.get_recommendation_diversity(hybrid_recs)
        assert 0 <= diversity <= 1, "Diversity score out of range"
        
        print("✅ Hybrid recommendations tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid recommendations test failed: {e}")
        return False

def test_ui():
    """Test UI modules."""
    print("🧪 Testing UI Modules...")
    
    try:
        from data_preprocessing import DataPreprocessor
        from content_based import ContentBasedRecommender
        from collaborative import CollaborativeRecommender
        from hybrid import HybridRecommender
        from ui import TerminalUI, StreamlitUI
        
        # Load and preprocess data
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data()
        cleaned_movies = preprocessor.clean_data()
        
        # Initialize recommenders
        content_recommender = ContentBasedRecommender(cleaned_movies)
        content_recommender.fit()
        
        collab_recommender = CollaborativeRecommender(data['ratings'], cleaned_movies)
        collab_recommender.prepare_data()
        collab_recommender.fit_svd()
        
        # Initialize hybrid recommender
        hybrid_recommender = HybridRecommender(content_recommender, collab_recommender)
        
        # Test TerminalUI initialization
        terminal_ui = TerminalUI(hybrid_recommender)
        assert terminal_ui is not None, "TerminalUI initialization failed"
        
        # Test StreamlitUI initialization (without running the app)
        try:
            streamlit_ui = StreamlitUI(hybrid_recommender)
            assert streamlit_ui is not None, "StreamlitUI initialization failed"
        except Exception as e:
            print(f"⚠️ StreamlitUI test skipped (Streamlit not available): {e}")
        
        print("✅ UI modules tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ UI modules test failed: {e}")
        return False

def test_main():
    """Test main module."""
    print("🧪 Testing Main Module...")
    
    try:
        from main import initialize_system, demo_mode
        
        # Test system initialization
        hybrid_recommender, data_summary = initialize_system()
        assert hybrid_recommender is not None, "System initialization failed"
        assert data_summary is not None, "No data summary generated"
        
        # Test demo mode
        demo_mode(hybrid_recommender)
        
        print("✅ Main module tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Main module test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting System Tests...")
    print("=" * 50)
    
    tests = [
        test_data_preprocessing,
        test_content_based,
        test_collaborative,
        test_hybrid,
        test_ui,
        test_main
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)