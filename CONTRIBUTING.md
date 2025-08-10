# Contributing to ASHES

Thank you for your interest in contributing to ASHES (Autonomous Scientific Hypothesis Evolution System)! We welcome contributions from researchers, developers, and scientists worldwide.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Development Environment Setup

1. **Fork the repository**

   ```bash
   git clone https://github.com/your-username/ashes.git
   cd ashes
   ```

2. **Set up Python environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Set up frontend environment**

   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Start development services**

   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Backend tests
pytest tests/ -v --cov=src/ashes

# Frontend tests
cd frontend && npm test

# Integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### 4. Code Quality Checks

```bash
# Python formatting and linting
black src/ tests/
flake8 src/ tests/
mypy src/

# Frontend formatting
cd frontend
npm run lint
npm run format
```

### 5. Commit Changes

Use conventional commit messages:

```bash
git commit -m "feat: add new hypothesis generation algorithm"
git commit -m "fix: resolve database connection timeout"
git commit -m "docs: update API documentation"
```

### 6. Submit Pull Request

- Push your branch to your fork
- Open a pull request against the main repository
- Fill out the PR template completely
- Link any relevant issues

## Contribution Areas

### 🤖 AI Agents

- Implement new agent types
- Improve existing agent reasoning
- Add domain-specific knowledge
- Enhance agent collaboration

### 🔬 Laboratory Integration

- Add support for new instruments
- Develop safety protocols
- Create experiment templates
- Improve automation workflows

### 📊 Data Management

- Enhance database schemas
- Implement new analytics
- Add visualization tools
- Improve search capabilities

### 🖥️ Frontend Development

- Build new UI components
- Enhance user experience
- Add interactive features
- Improve accessibility

### 🏗️ Infrastructure

- Optimize deployment processes
- Enhance monitoring systems
- Improve scalability
- Add security features

## Code Style Guidelines

### Python

- Use Black for formatting
- Follow PEP 8 guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Aim for 90%+ test coverage

Example:

```python
from typing import List, Optional
import asyncio

async def process_hypothesis(
    hypothesis: str, 
    context: Optional[Dict[str, Any]] = None
) -> HypothesisResult:
    """
    Process a scientific hypothesis using AI analysis.
    
    Args:
        hypothesis: The hypothesis statement to analyze
        context: Optional context for analysis
        
    Returns:
        HypothesisResult containing analysis and recommendations
        
    Raises:
        ValueError: If hypothesis is empty or invalid
    """
    if not hypothesis.strip():
        raise ValueError("Hypothesis cannot be empty")
    
    # Implementation here
    pass
```

### TypeScript/React

- Use TypeScript for all new code
- Follow React best practices
- Use functional components with hooks
- Write unit tests with Jest

Example:

```typescript
interface ExperimentProps {
  experimentId: string;
  onComplete: (result: ExperimentResult) => void;
}

const ExperimentComponent: React.FC<ExperimentProps> = ({
  experimentId,
  onComplete,
}) => {
  const [loading, setLoading] = useState(false);
  
  // Implementation here
  
  return (
    <div>
      {/* Component JSX */}
    </div>
  );
};
```

## Testing Guidelines

### Unit Tests

- Test individual functions and classes
- Mock external dependencies
- Use descriptive test names
- Aim for edge cases and error conditions

### Integration Tests

- Test component interactions
- Use real database connections
- Test API endpoints end-to-end
- Verify safety systems

### Performance Tests

- Benchmark critical algorithms
- Test under load conditions
- Monitor memory usage
- Verify scalability

## Documentation

### Code Documentation

- Use clear, concise docstrings
- Document all public APIs
- Include examples in docstrings
- Keep documentation up-to-date

### User Documentation

- Write clear tutorials
- Provide complete examples
- Include troubleshooting guides
- Document configuration options

## Security Considerations

### General Security

- Never commit API keys or secrets
- Use environment variables for configuration
- Validate all inputs
- Follow OWASP guidelines

### Laboratory Safety

- Implement emergency stop mechanisms
- Validate experimental protocols
- Monitor environmental conditions
- Log all safety events

## Release Process

### Version Numbering

We use Semantic Versioning (SemVer):

- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers bumped
- [ ] Security review completed
- [ ] Performance benchmarks run

## Getting Help

### Communication Channels

- **Discord**: [ASHES Community](https://discord.gg/ashes)
- **GitHub Discussions**: For design discussions
- **GitHub Issues**: For bug reports and feature requests
- **Email**: <chandu@portalvii.com>

### Mentorship

New contributors can request mentorship:

1. Comment on a "good first issue"
2. Join our Discord server
3. Attend weekly community calls
4. Pair programming sessions available

## Recognition

### Contributors

All contributors are recognized in:

- README.md contributors section
- Release notes
- Annual contributor report
- Conference presentations

### Contribution Types

We recognize various contribution types:

- 💻 Code contributions
- 📖 Documentation
- 🐛 Bug reports
- 💡 Ideas and feature requests
- 🎨 Design and UX
- 🧪 Testing and quality assurance
- 🌍 Translation and localization
- 📢 Community building

## License

By contributing to ASHES, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to the future of autonomous scientific research! 🚀
