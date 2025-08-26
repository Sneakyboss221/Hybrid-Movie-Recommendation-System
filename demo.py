"""
Simple Demo Script for Hybrid Movie Recommendation System

This script demonstrates the basic functionality of the system.
Run this to see the system in action without installing all dependencies.
"""

import os
import sys

def check_files():
    """Check if required files exist."""
    print("🔍 Checking required files...")
    
    required_files = ['movies.csv', 'ratings.csv']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        print("Please ensure movies.csv and ratings.csv are in the current directory.")
        return False
    
    return True

def show_project_structure():
    """Show the project structure."""
    print("\n📁 Project Structure:")
    print("=" * 40)
    
    files = [
        'data_preprocessing.py',
        'content_based.py', 
        'collaborative.py',
        'hybrid.py',
        'ui.py',
        'main.py',
        'streamlit_app.py',
        'requirements.txt',
        'README.md',
        'test_system.py',
        'demo.py'
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
    
    print(f"\n📊 Data Files:")
    if os.path.exists('movies.csv'):
        size = os.path.getsize('movies.csv') / 1024
        print(f"✅ movies.csv ({size:.1f} KB)")
    
    if os.path.exists('ratings.csv'):
        size = os.path.getsize('ratings.csv') / 1024 / 1024
        print(f"✅ ratings.csv ({size:.1f} MB)")

def show_usage_instructions():
    """Show usage instructions."""
    print("\n🚀 Usage Instructions:")
    print("=" * 40)
    
    print("\n1. Install Dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n2. Run Terminal Interface:")
    print("   python main.py")
    
    print("\n3. Run Web Interface:")
    print("   streamlit run streamlit_app.py")
    
    print("\n4. Run Demo Mode:")
    print("   python main.py --interface demo")
    
    print("\n5. Run Tests:")
    print("   python test_system.py")

def show_features():
    """Show system features."""
    print("\n🌟 System Features:")
    print("=" * 40)
    
    features = [
        "🎯 Content-Based Filtering (TF-IDF + Cosine Similarity)",
        "👥 Collaborative Filtering (SVD + NMF)",
        "🔗 Hybrid Recommendations (Weighted Combination)",
        "🎲 Ensemble Methods (Multi-Algorithm)",
        "❄️ Cold-Start Handling",
        "💡 Explainable Recommendations",
        "🎨 Serendipitous Recommendations",
        "📊 Recommendation Diversity",
        "🖥️ Terminal Interface",
        "🌐 Streamlit Web Interface",
        "📈 Interactive Visualizations",
        "🔍 Metadata Enrichment (TMDb API)",
        "💾 Model Persistence",
        "⚖️ Algorithm Comparison"
    ]
    
    for feature in features:
        print(f"  {feature}")

def show_example_code():
    """Show example code usage."""
    print("\n💻 Example Code Usage:")
    print("=" * 40)
    
    example_code = '''
# Basic usage example
from data_preprocessing import DataPreprocessor
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from hybrid import HybridRecommender

# Initialize system
preprocessor = DataPreprocessor()
data = preprocessor.load_data()
cleaned_movies = preprocessor.clean_data()

# Initialize recommenders
content_recommender = ContentBasedRecommender(cleaned_movies)
content_recommender.fit()

collab_recommender = CollaborativeRecommender(data['ratings'], cleaned_movies)
collab_recommender.prepare_data()
collab_recommender.fit_svd()

# Create hybrid recommender
hybrid_recommender = HybridRecommender(content_recommender, collab_recommender)

# Get recommendations
recommendations = hybrid_recommender.get_hybrid_recommendations(
    movie_title="Toy Story",
    user_id=1,
    num_recommendations=10,
    alpha=0.6  # 60% content-based, 40% collaborative
)

print("Top recommendations:")
for i, (_, row) in enumerate(recommendations.iterrows(), 1):
    print(f"{i}. {row['title']} - Score: {row['hybrid_score']:.3f}")
'''
    
    print(example_code)

def main():
    """Main demo function."""
    print("🎬 Hybrid Movie Recommendation System - Demo")
    print("=" * 50)
    
    # Check files
    if not check_files():
        print("\n❌ Cannot proceed without required data files.")
        return
    
    # Show project structure
    show_project_structure()
    
    # Show features
    show_features()
    
    # Show usage instructions
    show_usage_instructions()
    
    # Show example code
    show_example_code()
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed! The system is ready to use.")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the system: python main.py")
    print("3. Explore the web interface: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
