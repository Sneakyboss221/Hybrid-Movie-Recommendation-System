# Documentation

This directory contains the documentation for the Hybrid Movie Recommendation System.

## Structure

```
docs/
├── README.md              # This file
├── api/                   # API documentation
├── guides/                # User guides
├── tutorials/             # Tutorials and examples
└── _build/                # Built documentation (generated)
```

## Building Documentation

### Prerequisites
```bash
pip install -r requirements-dev.txt
```

### Build Commands
```bash
# Build HTML documentation
sphinx-build -b html . _build/html

# Build PDF documentation
sphinx-build -b latex . _build/latex

# Build all formats
make all
```

### View Documentation
After building, open `_build/html/index.html` in your browser.

## Documentation Guidelines

### Writing Style
- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Keep documentation up to date

### Code Examples
- Use Python code blocks with syntax highlighting
- Include complete, runnable examples
- Add expected output where helpful

### Screenshots
- Use high-quality screenshots
- Include alt text for accessibility
- Keep file sizes reasonable

## Contributing to Documentation

1. **Update existing docs** when changing code
2. **Add new docs** for new features
3. **Test examples** to ensure they work
4. **Follow the style guide** for consistency

## Documentation Tools

- **Sphinx**: Documentation generator
- **MyST**: Markdown support for Sphinx
- **Read the Docs**: Documentation hosting
- **GitHub Pages**: Alternative hosting

## Links

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [MyST Parser](https://myst-parser.readthedocs.io/)
- [Read the Docs](https://readthedocs.org/)
