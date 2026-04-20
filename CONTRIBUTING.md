# Contributing to EarthGuard - Asteroid Risk Prediction System

First off, thank you for considering contributing to EarthGuard! 🎉🚀

We welcome contributions from everyone. Here are the guidelines:

---

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Pull Request Process](#pull-request-process)
5. [Style Guidelines](#style-guidelines)
6. [Reporting Bugs](#reporting-bugs)
7. [Suggesting Enhancements](#suggesting-enhancements)

---

## Code of Conduct

This project and everyone participating in it is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

---

## How Can I Contribute?

### 🐛 Report a Bug
- Check if the bug has already been reported
- Use the bug report template
- Include detailed steps to reproduce

### 💡 Suggest an Enhancement
- Check if the feature already exists
- Describe the feature in detail
- Explain why it would be useful

### 🔧 Code Contributions
- Fix bugs
- Add new features
- Improve documentation
- Optimize performance

### 📚 Improve Documentation
- Fix typos
- Add examples
- Clarify instructions

---

## Development Setup

### Prerequisites
- Python 3.12+
- Git

### Setup Steps

```bash
# Fork the repository
# Then clone your fork
git clone https://github.com/your-username/EarthGuard.git
cd EarthGuard

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_web.txt

# Run the app
streamlit run app.py