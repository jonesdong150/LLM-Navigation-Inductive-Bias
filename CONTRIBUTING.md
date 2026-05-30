# Contributing

Thank you for your interest in contributing to this project.

## Development Setup

```bash
git clone https://github.com/yourusername/navigation-planning-llm.git
cd navigation-planning-llm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"   # Install dev tools (pytest, black, flake8, isort)
```

## Running Tests

```bash
# Run all tests
python tests/test_scene_serializer.py
python tests/test_generate_dataset.py

# Or with pytest
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to public functions
- Write unit tests for new functionality

```bash
# Format
black .
isort .

# Lint
flake8 .
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest tests/`
4. Validate dataset: `python tools/validate_dataset.py --validate`
5. Commit: `git commit -m "Add: description"`
6. Push and create PR

## Extending the Dataset

### Adding New Room/Object Types

Edit `tools/knowledge_base.py`:
- Add entries to `ROOM_CATEGORIES` or `OBJECT_CATEGORIES`
- Ensure each entry has canonical name, at least 2 synonyms, and abbreviation

### Adding New Query Types

1. Add generation logic in `tools/generate_dataset.py`
2. Add verification logic in `tools/verify_queries.py`
3. Update `tools/validate_dataset.py` if needed
4. Add tests

### Adding New Format Variants

1. Add serialization method in `tools/scene_serializer.py`
2. Register in `generate_all_variants()` or `generate_dimension_variants()`
3. Update `tools/validate_dataset.py` expected variants
4. Add tests

## Bug Reports

Please include:
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Error traceback (if applicable)

## Questions

Open a GitHub issue or use Discussions.
