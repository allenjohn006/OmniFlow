# 📚 Documentation Overview

This project now includes comprehensive documentation to help users understand, deploy, and contribute to OmniFlow Sales AI.

## 📖 Documentation Files

All documentation is located in the `/docs` directory:

### 1. **[01-ARCHITECTURE.md](01-ARCHITECTURE.md)**
**Purpose**: System design and component overview  
**Contents**:
- High-level system architecture
- Component breakdown (FastAPI, Django, ML Pipeline)
- Data flow diagrams
- Key design patterns (async jobs, guardrail logic, dtype safety)
- Scalability considerations
- Security recommendations

**Best for**: Understanding how the system works internally

---

### 2. **[02-GETTING_STARTED.md](02-GETTING_STARTED.md)**
**Purpose**: Setup and quick start guide  
**Contents**:
- Prerequisites and installation steps
- Virtual environment setup
- Running both via dev.py or individual commands
- Data preparation
- Accessing the application
- Troubleshooting common issues

**Best for**: Getting the system up and running quickly

---

### 3. **[03-API_REFERENCE.md](03-API_REFERENCE.md)**
**Purpose**: Complete API endpoint documentation  
**Contents**:
- Base URL and authentication
- All endpoints (training, drift, prediction)
- Request/response formats with examples
- Status codes and error handling
- Data format specifications
- Example workflows (Python + curl)

**Best for**: Integrating with the API or building client applications

---

### 4. **[04-ML_PIPELINE.md](04-ML_PIPELINE.md)**
**Purpose**: ML algorithm details and methodology  
**Contents**:
- Data lifecycle (ingestion → preprocessing → training)
- Feature engineering (15 engineered features)
- Model architecture (scikit-learn Pipeline + XGBoost)
- Metrics calculations (R², MAE, RMSE)
- Drift detection algorithms (KS test, Chi-square)
- Auto-retrain decision logic (guardrails)
- Performance optimization tips

**Best for**: Data scientists and ML engineers

---

### 5. **[05-CONTRIBUTING.md](05-CONTRIBUTING.md)**
**Purpose**: Contributing guidelines for developers  
**Contents**:
- Fork and setup instructions
- Code standards (PEP 8, black, flake8)
- Git commit conventions
- Testing requirements
- PR process and templates
- Code review checklist
- Debugging tips

**Best for**: Contributors and developers

---

### 6. **[06-DEPLOYMENT.md](06-DEPLOYMENT.md)**
**Purpose**: Production deployment guide  
**Contents**:
- Docker containerization (Dockerfile, docker-compose)
- Cloud deployment options (AWS EC2, Google Cloud Run, Heroku)
- Environment variables and configuration
- Security hardening checklist
- Monitoring and logging setup
- Scaling strategies (vertical and horizontal)
- Disaster recovery and backups
- Troubleshooting production issues

**Best for**: DevOps engineers and operations teams

---

## 🗺️ Learning Path

**New to the project?** Follow this order:

1. 📖 **Read first**: [README.md](../README.md) - Project overview
2. 🏗️ **Understand**: [01-ARCHITECTURE.md](01-ARCHITECTURE.md) - How it works
3. 🚀 **Get started**: [02-GETTING_STARTED.md](02-GETTING_STARTED.md) - Installation & running
4. 🔌 **Try the API**: [03-API_REFERENCE.md](03-API_REFERENCE.md) - Explore endpoints
5. 🧠 **Deep dive**: [04-ML_PIPELINE.md](04-ML_PIPELINE.md) - ML details
6. 🤝 **Contribute**: [05-CONTRIBUTING.md](05-CONTRIBUTING.md) - Start coding
7. 🚀 **Deploy**: [06-DEPLOYMENT.md](06-DEPLOYMENT.md) - Production ready

---

## 📊 Document Statistics

| Document | Words | Sections | Code Examples |
|----------|-------|----------|----------------|
| 01-ARCHITECTURE.md | ~3,500 | 12 | 8 |
| 02-GETTING_STARTED.md | ~2,800 | 10 | 15 |
| 03-API_REFERENCE.md | ~3,200 | 15 | 25 |
| 04-ML_PIPELINE.md | ~4,100 | 14 | 12 |
| 05-CONTRIBUTING.md | ~4,000 | 18 | 20 |
| 06-DEPLOYMENT.md | ~3,800 | 16 | 30 |
| **Total** | **~21,400** | **85** | **110** |

---

## 🔍 Quick Reference

### For Users
- Want to use the system? → [02-GETTING_STARTED.md](02-GETTING_STARTED.md)
- Want to make an API call? → [03-API_REFERENCE.md](03-API_REFERENCE.md)
- Got an issue? → [02-GETTING_STARTED.md § Troubleshooting](02-GETTING_STARTED.md#troubleshooting)

### For Data Scientists
- How does drift detection work? → [04-ML_PIPELINE.md § Drift Detection](04-ML_PIPELINE.md#drift-detection)
- What metrics are calculated? → [04-ML_PIPELINE.md § Metrics Calculation](04-ML_PIPELINE.md#metrics-calculation)
- How is retraining decided? → [04-ML_PIPELINE.md § Auto-Retrain Decision Logic](04-ML_PIPELINE.md#auto-retrain-decision-logic)

### For Developers
- Code standards? → [05-CONTRIBUTING.md § Code Standards](05-CONTRIBUTING.md#code-standards)
- How to write tests? → [05-CONTRIBUTING.md § Testing](05-CONTRIBUTING.md#testing)
- PR requirements? → [05-CONTRIBUTING.md § Pull Request Process](05-CONTRIBUTING.md#pull-request-process)

### For DevOps
- Docker setup? → [06-DEPLOYMENT.md § Docker Containerization](06-DEPLOYMENT.md#2-docker-containerization)
- AWS deployment? → [06-DEPLOYMENT.md § AWS EC2](06-DEPLOYMENT.md#aws-ec2)
- Production setup? → [06-DEPLOYMENT.md § Production Configuration](06-DEPLOYMENT.md#production-configuration)

---

## 🎯 Key Documentation Features

✅ **Comprehensive**: 21,400+ words covering all aspects  
✅ **Practical**: 110+ code examples and real-world scenarios  
✅ **Accessible**: Written for different technical levels  
✅ **Complete**: Architecture, API, ML, deployment, contributing  
✅ **Up-to-date**: Reflects current codebase state (Apr 2026)  
✅ **Navigable**: TOC in each file, cross-references between docs  

---

## 📝 Contributing to Documentation

Found an error or want to improve documentation?

1. Edit the relevant `.md` file in `/docs`
2. Follow Markdown formatting conventions
3. Update table of contents if adding sections
4. Submit PR with changes

**Documentation PR Template**:
```markdown
**Type**: docs
**File**: 03-API_REFERENCE.md
**Change**: Added Python client example for /predict endpoint
**Why**: Users asked for Python integration examples
```

See [05-CONTRIBUTING.md](05-CONTRIBUTING.md) for full guidelines.

---

## 🔗 Related Files

- **[README.md](../README.md)** - Project overview (intro point)
- **[CHANGELOG.md](../CHANGELOG.md)** - Version history
- **.gitignore** - Git ignore rules
- **[requirements.txt](../requirements.txt)** - Python dependencies

---

## 📞 Support

**Questions about documentation?**
- 🐛 Found a typo? → Create an issue
- 💡 Need clarification? → Create a discussion
- ✨ Have a suggestion? → Create an enhancement issue

---

**Last Updated**: April 2026  
**Maintainer**: OmniFlow Team
