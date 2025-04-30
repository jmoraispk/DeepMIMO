# Contributing Guide

Thank you for your interest in contributing to DeepMIMO! This guide will help you get started.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DeepMIMO.git
   cd DeepMIMO
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use Google-style docstrings
- Sort imports using `isort`

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## Documentation

Build documentation locally:
```bash
cd docs
make html
```

## Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Any related issues
   - Test coverage
   - Documentation updates

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. 