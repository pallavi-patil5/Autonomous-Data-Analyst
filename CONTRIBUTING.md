# Contributing to Autonomous Data Science AI

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes thoroughly
6. Commit with clear messages: `git commit -m "Add: feature description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## 🧩 Adding New Agents

To add a new agent to the system:

1. **Create Agent Class**
   - Inherit from `BaseAgent`
   - Place in a new file: `enhanced_<agent_name>_agent.py`

2. **Implement Required Methods**
   ```python
   from base_agent import BaseAgent
   
   class YourAgent(BaseAgent):
       def __init__(self, data=None):
           super().__init__()
           self.data = data
       
       def get_capabilities(self):
           return [
               {
                   "function_name": "your_function",
                   "description": "Clear description",
                   "parameters": ["param1", "param2"],
                   "examples": ["Example query 1", "Example query 2"]
               }
           ]
       
       def your_function(self, param1, param2):
           # Implementation
           pass
   ```

3. **Register in Orchestrator**
   - Import in `smart_orchestrator_v2.py`
   - Add to `_initialize_agents()` or `set_data()`

## 📝 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to all functions
- Keep functions focused and concise
- Add type hints where appropriate

## 🧪 Testing

- Test your agent independently before integration
- Verify with multiple query types
- Check error handling
- Ensure no breaking changes to existing functionality

## 📋 Pull Request Guidelines

- Provide clear description of changes
- Reference any related issues
- Include examples of new functionality
- Update documentation if needed
- Ensure all tests pass

## 🐛 Reporting Bugs

When reporting bugs, include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)
- Error messages/logs

## 💡 Feature Requests

- Check if feature already exists
- Provide clear use case
- Explain expected behavior
- Consider implementation approach

## 📞 Questions?

Open an issue with the "question" label for any clarifications.

---

Thank you for contributing! 🎉
