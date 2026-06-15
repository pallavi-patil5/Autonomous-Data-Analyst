# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please follow these steps:

1. **Do NOT** open a public issue
2. Email the details to: [your-security-email@example.com]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Best Practices

### API Key Management
- Never commit API keys to version control
- Use environment variables or secure vaults
- Rotate keys regularly
- Use `.env` files (excluded in `.gitignore`)

### Data Privacy
- All data processing happens locally
- No data is stored permanently by the application
- Only query text is sent to Gemini API for processing
- Uploaded datasets remain in memory only

### Dependencies
- Regularly update dependencies: `pip install --upgrade -r requirements.txt`
- Monitor security advisories for used packages
- Use virtual environments to isolate dependencies

### Streamlit Deployment
- Use HTTPS in production
- Enable authentication if deploying publicly
- Set appropriate CORS policies
- Use Streamlit secrets management for API keys

## Known Security Considerations

1. **API Key Exposure**: Ensure API keys are never logged or displayed
2. **File Upload**: Only CSV/Excel files are accepted; validate file types
3. **Code Execution**: Dynamic capability execution is sandboxed within agent methods
4. **LLM Injection**: User queries are processed but not executed as code directly

## Updates

Security updates will be released as patch versions and documented in CHANGELOG.md.
