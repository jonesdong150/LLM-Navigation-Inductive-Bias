# Contributing to Navigation Planning LLM Research

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## 🎯 Ways to Contribute

- **Bug Reports**: Submit issues for bugs or unexpected behavior
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Improve documentation, fix typos, or add examples
- **Dataset Extensions**: Add new scene types or query formats

## 🐛 Reporting Bugs

Before submitting a bug report, please:

1. Check if the issue has already been reported
2. Use the latest version of the code
3. Provide a minimal reproducible example

**Bug Report Template**:

```markdown
## Description
Brief description of the bug

## Steps to Reproduce
1. Run command X
2. With parameters Y
3. Error occurs

## Expected Behavior
What should happen

## Actual Behavior
What actually happened

## Environment
- Python version:
- Operating System:
- Package versions:
```

## 🔧 Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/navigation-planning-llm.git
cd navigation-planning-llm

# Create development environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
pytest tests/
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Write unit tests for new functionality

### Code Formatting

We use the following tools:
- `black` for code formatting
- `flake8` for linting
- `isort` for import sorting

```bash
# Format code
black .

# Check linting
flake8 .

# Sort imports
isort .
```

## 🧪 Testing

Before submitting a pull request:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=tools tests/

# Validate dataset
python tools/validate_dataset.py --validate
```

## 📦 Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Run Tests and Validation**
   ```bash
   pytest tests/
   python tools/validate_dataset.py --validate
   ```

4. **Commit Your Changes**
   ```bash
   git commit -m "Add: brief description of your changes"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Dataset extension

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Documentation updated

## Related Issues
Fixes #issue_number
```

## 📊 Dataset Contributions

If you want to add new scene types or query formats:

1. Follow the existing structure in `data_set/`
2. Implement serialization in `tools/scene_serializer.py`
3. Add generation logic to `tools/generate_dataset.py`
4. Ensure information equivalence across all variants
5. Add validation tests

### Scene JSON Schema

```json
{
  "scene_id": "string",
  "description": "string",
  "world": {
    "scene_name": "string",
    "rooms": [
      {
        "idx": int,
        "room_id": "string",
        "type": "string",
        "x": int,
        "y": int
      }
    ],
    "edges": [[int, int]],
    "objects": {int: [[string, string]]},
    "history": [int],
    "rules": [string]
  },
  "variants": {
    "flat": "string",
    "hier": "string",
    ...
  }
}
```

## ❓ Questions and Support

- Open an issue for bugs or feature requests
- Use GitHub Discussions for general questions
- Email maintainers for sensitive issues

## 📜 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Academic paper acknowledgments (for significant contributions)

Thank you for contributing to making LLM spatial reasoning research more transparent and reproducible!
