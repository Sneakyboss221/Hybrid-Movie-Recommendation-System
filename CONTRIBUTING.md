# Contributing to Hybrid Movie Recommendation System

Thank you for your interest in contributing to our project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug Reports**: Report issues you find
- 💡 **Feature Requests**: Suggest new features
- 📝 **Documentation**: Improve docs and examples
- 🔧 **Code**: Submit pull requests
- 🧪 **Testing**: Add or improve tests
- 🌟 **Examples**: Create usage examples

### Before You Start

1. **Check existing issues** to avoid duplicates
2. **Read the documentation** to understand the project
3. **Set up your development environment** (see below)

## 🛠️ Development Setup

### Prerequisites
- Python 3.7+
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Fork the repository**
   ```bash
   # Fork on GitHub first, then clone your fork
   git clone https://github.com/YOUR_USERNAME/hybrid-movie-recommendation-system.git
   cd hybrid-movie-recommendation-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run tests**
   ```bash
   python test_system.py
   ```

## 📝 Code Style Guidelines

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints where appropriate
- Keep functions under 50 lines
- Use descriptive variable names

### Documentation
- Add docstrings to all functions and classes
- Use Google-style docstrings
- Include examples in docstrings
- Update README.md for new features

### Example Code Style
```python
def get_recommendations(
    movie_title: str, 
    num_recommendations: int = 10
) -> Optional[pd.DataFrame]:
    """
    Get movie recommendations based on content similarity.
    
    Args:
        movie_title: Title of the movie to find similar movies for
        num_recommendations: Number of recommendations to return
        
    Returns:
        DataFrame with recommendations or None if movie not found
        
    Example:
        >>> recs = get_recommendations("Toy Story", 5)
        >>> print(recs['title'].tolist())
        ['Monsters, Inc.', 'Finding Nemo', ...]
    """
    # Implementation here
    pass
```

## 🔄 Pull Request Process

### Before Submitting

1. **Test your changes**
   ```bash
   python test_system.py
   python -m pytest tests/  # If you have pytest
   ```

2. **Check code style**
   ```bash
   flake8 .
   black .  # Auto-format code
   ```

3. **Update documentation**
   - Update README.md if needed
   - Add docstrings to new functions
   - Update any relevant examples

### Pull Request Template

Use this template when creating a PR:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Test addition
- [ ] Other (please describe)

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes
- [ ] Self-review completed

## Screenshots (if applicable)
Add screenshots for UI changes
```

## 🐛 Bug Reports

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 10, macOS 12]
- Python Version: [e.g., 3.9.7]
- Package Versions: [from pip freeze]

## Additional Information
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why this feature would be useful

## Proposed Implementation
How you think it could be implemented

## Alternatives Considered
Other approaches you've considered

## Additional Information
Any other relevant information
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
python test_system.py

# Run specific test file
python -m pytest tests/test_content_based.py

# Run with coverage
python -m pytest --cov=. tests/
```

### Writing Tests
- Test both success and failure cases
- Use descriptive test names
- Mock external dependencies
- Test edge cases

### Example Test
```python
def test_get_recommendations_valid_movie():
    """Test getting recommendations for a valid movie."""
    recommender = ContentBasedRecommender(movies_df)
    recommender.fit()
    
    recommendations = recommender.get_recommendations("Toy Story", 5)
    
    assert recommendations is not None
    assert len(recommendations) == 5
    assert "Toy Story" not in recommendations['title'].values

def test_get_recommendations_invalid_movie():
    """Test getting recommendations for an invalid movie."""
    recommender = ContentBasedRecommender(movies_df)
    recommender.fit()
    
    recommendations = recommender.get_recommendations("NonExistentMovie", 5)
    
    assert recommendations is None
```

## 📚 Documentation Guidelines

### Code Documentation
- Use Google-style docstrings
- Include type hints
- Provide examples
- Document exceptions

### README Updates
- Update installation instructions
- Add usage examples
- Update feature list
- Include screenshots for UI changes

## 🏷️ Commit Message Guidelines

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

### Examples
```
feat(hybrid): add ensemble recommendation method

fix(ui): resolve duplicate button ID error

docs(readme): add installation instructions

test(content): add edge case tests for empty dataset
```

## 🎯 Project Structure

Understanding the project structure helps with contributions:

```
hybrid-movie-recommendation-system/
├── 📁 Core Modules
│   ├── data_preprocessing.py      # Data loading and cleaning
│   ├── content_based.py          # Content-based filtering
│   ├── collaborative.py          # Collaborative filtering
│   ├── hybrid.py                 # Hybrid recommendation engine
│   └── ui.py                     # User interface components
├── 🌐 Web Applications
│   ├── streamlit_app.py          # Streamlit web interface
│   └── main.py                   # Main application entry
├── 📊 Data
│   ├── movies.csv                # Movie metadata
│   └── ratings.csv               # User ratings
├── 🧪 Testing & Documentation
│   ├── test_system.py            # System testing suite
│   ├── demo.py                   # Quick demo script
│   └── README.md                 # Main documentation
└── 📋 Configuration
    ├── requirements.txt          # Python dependencies
    └── .gitignore               # Git ignore rules
```

## 🤝 Community Guidelines

### Be Respectful
- Be kind and respectful to all contributors
- Use inclusive language
- Welcome newcomers

### Communication
- Ask questions if something is unclear
- Provide constructive feedback
- Help others learn

### Code Review
- Review others' code constructively
- Suggest improvements politely
- Focus on the code, not the person

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/hybrid-movie-recommendation-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/hybrid-movie-recommendation-system/discussions)
- **Email**: your.email@example.com

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Hybrid Movie Recommendation System! 🎬✨
