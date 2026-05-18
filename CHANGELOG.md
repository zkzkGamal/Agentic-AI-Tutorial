# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Automated Testing & CI Pipeline**: Introduced GitHub Actions CI workflow (`.github/workflows/chapter5-ci.yml`) to automatically run tests across multiple Python versions (3.10 and 3.11).
- **Real-Time Pytest Execution**: Customized `pytest` via `conftest.py` in Chapter 5 to provide instant, unbuffered terminal feedback for test collection and test steps.
- **Unified Requirements**: Created a root `requirements.txt` file that elegantly consolidates all chapter-specific dependencies with proper documentation and comments, allowing for a single-command setup.
- **Documentation Demos**: Added demo videos/images to the README showcasing the real-time execution of pytest.
- **Articles Featured Pass**: Added relevant article features and links to the project structure.

### Changed
- **Main README.md**: Updated Quick Start instructions to feature the new root `requirements.txt`. Expanded the Chapter 5 section to emphasize the critical role of automated testing in Agentic workflows.
- **Chapter 5 README.md**: Deeply expanded the automated testing section to document the real-time pytest feedback mechanism and the new GitHub Actions pipeline.

