# Contributing to PolyB0T

Thank you for your interest in contributing to PolyB0T! This document provides guidelines for contributions.

## Code of Conduct

- Be respectful and constructive
- Focus on educational and research purposes
- No promotion of illegal activities or ToS violations
- Maintain high code quality standards

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/polyb0t.git
   cd polyb0t
   ```

2. **Install development dependencies**
   ```bash
   poetry install --with dev
   poetry shell
   ```

3. **Set up pre-commit hooks (optional but recommended)**
   ```bash
   poetry run pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add type hints everywhere
   - Include docstrings for public methods

3. **Write tests**
   - Add tests for new functionality
   - Ensure existing tests pass
   - Aim for >80% code coverage

4. **Run quality checks**
   ```bash
   # Type checking
   poetry run mypy polyb0t

   # Linting
   poetry run ruff check polyb0t

   # Formatting
   poetry run black polyb0t

   # Tests
   poetry run pytest -v --cov=polyb0t
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Adding or updating tests
   - `refactor:` Code refactoring
   - `perf:` Performance improvements

6. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Testing Guidelines

- Write unit tests for individual components
- Use fixtures for common test data (see `tests/conftest.py`)
- Mock external API calls
- Test edge cases and error conditions
- Tests should be deterministic (no random failures)

Example test structure:
```python
def test_feature_name():
    """Test description."""
    # Arrange
    setup_data = ...

    # Act
    result = function_under_test(setup_data)

    # Assert
    assert result == expected_value
```

## Code Style

- Follow PEP 8
- Use type hints for all function parameters and return values
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all public classes and methods

Example docstring format:
```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of function.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When and why this is raised.
    """
    pass
```

## Areas for Contribution

### High Priority
- [ ] Improve strategy with additional features
- [ ] Add more sophisticated position sizing (Kelly criterion)
- [ ] Implement correlation tracking between markets
- [ ] Add more comprehensive backtesting
- [ ] Improve fill simulation realism
- [ ] Add performance analytics dashboard

### Medium Priority
- [ ] ML-based probability estimation
- [ ] Sentiment analysis integration
- [ ] Multi-market arbitrage detection
- [ ] Advanced risk metrics (Sharpe, Sortino)
- [ ] Real-time dashboard (web UI)
- [ ] Alert system (email, Telegram)

### Documentation
- [ ] More examples and tutorials
- [ ] Strategy development guide
- [ ] API documentation
- [ ] Video tutorials

### Testing
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Stress testing
- [ ] Historical backtests

## Architecture Guidelines

When adding new features:

1. **Maintain separation of concerns**
   - Data layer: API clients, database
   - Models: Strategy, features, risk
   - Execution: Orders, portfolio, simulator
   - Services: Orchestration, reporting
   - API/CLI: User interfaces

2. **Keep components testable**
   - Inject dependencies
   - Use interfaces/protocols
   - Avoid global state
   - Mock external dependencies

3. **Follow existing patterns**
   - Look at similar existing code
   - Maintain consistent style
   - Use established utilities

4. **Document assumptions**
   - Comment non-obvious logic
   - Document API response formats
   - Explain algorithm choices

## Safety and Compliance

**Critical**: Any contributions must maintain safety-by-default:

- ❌ Do NOT implement features that bypass geo-restrictions
- ❌ Do NOT add live trading without explicit user confirmation
- ❌ Do NOT remove or weaken risk checks
- ❌ Do NOT commit API keys or credentials
- ✅ DO maintain paper trading as default
- ✅ DO add appropriate warnings
- ✅ DO respect platform ToS

## Review Process

Pull requests will be reviewed for:
- Code quality and style
- Test coverage
- Documentation
- Safety considerations
- Performance impact
- Compatibility

Reviewers may request changes. Please respond promptly and be open to feedback.

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for general questions
- Tag maintainers for urgent matters

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make PolyB0T better! 🚀

