# 🎬 Hybrid Movie Recommendation System

A powerful, modular hybrid movie recommendation engine that combines **Content-Based Filtering** and **Collaborative Filtering** techniques using advanced machine learning algorithms.

## 🌟 Features

### 🧠 Core Algorithms
- **Content-Based Filtering**: TF-IDF vectorization with cosine similarity
- **Collaborative Filtering**: SVD (Singular Value Decomposition) and NMF (Non-negative Matrix Factorization)
- **Hybrid Recommendations**: Weighted combination of both approaches
- **Ensemble Methods**: Multi-algorithm recommendations with customizable weights

### 🎯 Advanced Features
- **Cold-Start Handling**: Automatic fallback for new users/movies
- **Explainable Recommendations**: Understand why movies are recommended
- **Serendipitous Recommendations**: Balance similarity with novelty
- **Recommendation Diversity**: Measure and optimize recommendation variety
- **Metadata Enrichment**: Optional TMDb API integration for enhanced movie data

### 🖥️ User Interfaces
- **Terminal Interface**: Interactive command-line interface
- **Streamlit Web App**: Modern, responsive web interface with visualizations
- **API-Ready**: Modular design for easy integration

### 📊 Analytics & Visualization
- **System Statistics**: Comprehensive data insights
- **User Profiles**: Detailed user preference analysis
- **Algorithm Comparison**: Side-by-side performance evaluation
- **Interactive Charts**: Plotly-based visualizations

## 🏗️ Architecture

```
├── data_preprocessing.py    # Data loading, cleaning, and enrichment
├── content_based.py        # TF-IDF and content-based filtering
├── collaborative.py        # SVD/NMF collaborative filtering
├── hybrid.py              # Hybrid recommendation logic
├── ui.py                  # Terminal and Streamlit interfaces
├── main.py                # Main application entry point
├── streamlit_app.py       # Standalone Streamlit web app
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd hybrid-movie-recommendation-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

The system uses the MovieLens 100k dataset. Ensure you have:
- `movies.csv` - Movie metadata (movieId, title, genres)
- `ratings.csv` - User ratings (userId, movieId, rating, timestamp)

### 3. Basic Usage

#### Terminal Interface
```bash
# Run with terminal interface (default)
python main.py

# Run demo mode
python main.py --interface demo

# Run with metadata enrichment (requires TMDb API key)
python main.py --enrich-metadata --tmdb-api-key YOUR_API_KEY
```

#### Web Interface
```bash
# Run Streamlit web app
streamlit run streamlit_app.py

# Or use main.py
python main.py --interface streamlit
```

## 📖 Detailed Usage

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --interface {terminal,streamlit,demo}
                        Interface to use (default: terminal)
  --enrich-metadata     Enable metadata enrichment with TMDb API
  --tmdb-api-key TEXT   TMDb API key for metadata enrichment
  --save-models         Save trained models to disk
  --load-models         Load trained models from disk
```

### Programmatic Usage

```python
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
```

## 🎯 Core Functions

### Content-Based Filtering
```python
# Get similar movies based on content
content_recs = content_recommender.get_recommendations("Toy Story", num_recommendations=10)

# Get movies by genre
genre_recs = content_recommender.get_similar_movies_by_genre("Animation", num_recommendations=10)

# Explain recommendations
explanation = content_recommender.explain_recommendation("Toy Story", "Monsters, Inc.")
```

### Collaborative Filtering
```python
# Get user recommendations
user_recs = collab_recommender.get_user_recommendations(user_id=1, num_recommendations=10)

# Get user profile
profile = collab_recommender.get_user_profile(user_id=1)

# Evaluate model performance
metrics = collab_recommender.evaluate_model('svd')
```

### Hybrid Recommendations
```python
# Basic hybrid recommendations
hybrid_recs = hybrid_recommender.get_hybrid_recommendations(
    movie_title="Toy Story",
    user_id=1,
    num_recommendations=10,
    alpha=0.5
)

# Ensemble recommendations
ensemble_recs = hybrid_recommender.get_ensemble_recommendations(
    movie_title="Toy Story",
    user_id=1,
    num_recommendations=10,
    weights={'content': 0.4, 'collaborative': 0.4, 'popularity': 0.2}
)

# Serendipitous recommendations
serendipitous_recs = hybrid_recommender.get_serendipitous_recommendations(
    movie_title="Toy Story",
    user_id=1,
    num_recommendations=10,
    serendipity_weight=0.3
)
```

## 🔧 Configuration

### TMDb API Setup (Optional)
For enhanced metadata (plot summaries, cast, posters):

1. Get a free API key from [TMDb](https://www.themoviedb.org/settings/api)
2. Use with the system:
   ```bash
   python main.py --enrich-metadata --tmdb-api-key YOUR_API_KEY
   ```

### Model Persistence
Save and load trained models for faster startup:

```bash
# Save models
python main.py --save-models

# Load models (faster startup)
python main.py --load-models
```

## 📊 Performance & Evaluation

### Metrics
- **RMSE**: Root Mean Square Error for rating predictions
- **MAE**: Mean Absolute Error for rating predictions
- **Diversity Score**: Measure of recommendation variety (0-1)
- **Serendipity Score**: Balance between similarity and novelty

### Typical Performance
- **Content-Based**: High precision, good for cold-start
- **Collaborative**: Better personalization, requires user history
- **Hybrid**: Best overall performance, handles cold-start
- **Ensemble**: Most robust, combines multiple approaches

## 🎨 User Interfaces

### Terminal Interface
Interactive menu-driven interface with options:
1. Get movie recommendations by title
2. Get recommendations for a user
3. Compare different algorithms
4. View user profile
5. Get serendipitous recommendations
6. View system statistics

### Streamlit Web Interface
Modern web app with:
- **Movie Recommendations**: Search and get recommendations
- **User Recommendations**: Personalized recommendations
- **Algorithm Comparison**: Side-by-side algorithm evaluation
- **Analytics**: Data insights and visualizations
- **Serendipitous**: Novel recommendation discovery

## 🔍 Example Output

### Movie Recommendations
```
🎯 HYBRID RECOMMENDATIONS (α=0.6):
------------------------------------------------------------

 1. Monsters, Inc. (2001)
    Genres: Adventure|Animation|Children|Comedy|Fantasy
    Hybrid Score: 0.847

 2. Finding Nemo (2003)
    Genres: Adventure|Animation|Children|Comedy
    Hybrid Score: 0.823

 3. The Incredibles (2004)
    Genres: Action|Adventure|Animation|Children|Comedy
    Hybrid Score: 0.798

💡 EXPLANATION: 'Monsters, Inc.' was recommended because it shares these features with 'Toy Story': 'animation', 'children', 'comedy', 'adventure', 'fantasy'

🎯 RECOMMENDATION DIVERSITY: 0.734
```

## 🛠️ Development

### Project Structure
```
├── models/                 # Saved model files
├── data/                   # Data files (movies.csv, ratings.csv)
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

### Adding New Algorithms
The modular design makes it easy to add new recommendation algorithms:

1. Create a new recommender class
2. Implement the required interface methods
3. Integrate with the hybrid recommender
4. Add to the UI interfaces

### Testing
```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=.
```

## 📈 Future Enhancements

- [ ] Deep learning embeddings (Word2Vec, BERT)
- [ ] Real-time recommendation updates
- [ ] Multi-modal recommendations (posters, trailers)
- [ ] A/B testing framework
- [ ] Recommendation explanation visualization
- [ ] User feedback integration
- [ ] Scalable deployment (Docker, Kubernetes)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MovieLens**: For the excellent dataset
- **TMDb**: For movie metadata API
- **Surprise**: For collaborative filtering algorithms
- **Streamlit**: For the web interface framework

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review example notebooks

---

**Made with ❤️ for the movie recommendation community**



