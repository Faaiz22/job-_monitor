# Job Monitoring System 💼

A comprehensive Python-based job monitoring system that scrapes job postings from multiple sites, uses machine learning for categorization and skill extraction, matches jobs to user preferences, and sends personalized email notifications with an interactive Streamlit dashboard.

## 🚀 Features

- **Multi-Site Web Scraping**: Automated scraping from Indeed, LinkedIn, and Glassdoor
- **ML-Powered Categorization**: Intelligent job classification and skill extraction using NLP
- **Smart Matching**: Algorithm-based job matching to user profiles and preferences
- **Email Notifications**: Personalized job alerts with customizable frequency
- **Interactive Dashboard**: Real-time analytics and job browsing via Streamlit
- **Scalable Architecture**: Modular design with proper error handling and logging

## 📁 Project Structure

```
job-monitoring-system/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── __init__.py
│   └── settings.py              # All configurations and URLs
├── src/
│   ├── scraper/                 # Web scraping modules
│   ├── preprocessing/           # Data cleaning and validation
│   ├── ml_categorizer/          # ML models for categorization
│   ├── matching_notifier/       # Job matching and notifications
│   └── dashboard/               # Streamlit web interface
├── data/
│   ├── skills_dictionary.json  # Comprehensive skills database
│   ├── user_profiles.json       # User preferences and profiles
│   ├── raw/                     # Raw scraped data
│   └── processed/               # Cleaned data
├── models/                      # Trained ML models
├── logs/                        # Application logs
└── tests/                       # Unit tests
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Chrome browser (for Selenium)
- ChromeDriver

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/job-monitoring-system.git
   cd job-monitoring-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

6. **Initialize directories**
   ```bash
   mkdir -p data/raw data/processed models/trained_models logs
   ```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Email Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password

# Scraping Settings
SCRAPER_DELAY_MIN=2
SCRAPER_DELAY_MAX=5
MAX_PAGES_PER_SITE=10

# Machine Learning
SPACY_MODEL=en_core_web_sm
MIN_CONFIDENCE_THRESHOLD=0.7

# Dashboard
STREAMLIT_PORT=8501
```

### User Profiles

Edit `data/user_profiles.json` to add users:

```json
{
  "users": [
    {
      "id": 1,
      "name": "John Doe",
      "email": "john@example.com",
      "active": true,
      "preferences": {
        "job_categories": ["Software Development"],
        "experience_level": "mid_level",
        "job_types": ["Full-time", "Remote"],
        "locations": ["Remote", "San Francisco, CA"],
        "salary_range": {"min": 80000, "max": 150000}
      },
      "skills": {
        "required": ["Python", "JavaScript", "React"],
        "preferred": ["AWS", "Docker", "Kubernetes"]
      }
    }
  ]
}
```

## 🚀 Usage

### Command Line Interface

```bash
# Run the complete pipeline
python main.py

# Run individual components
python -m src.scraper.indeed_scraper
python -m src.ml_categorizer.model_trainer
python -m src.matching_notifier.job_matcher
```

### Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
```

Access the dashboard at `http://localhost:8501`

### Scheduled Execution

Use the built-in scheduler or set up cron jobs:

```bash
# Run daily at 9 AM
0 9 * * * /path/to/venv/bin/python /path/to/project/main.py
```

## 🧠 Machine Learning Pipeline

### 1. Skill Extraction
- Uses spaCy NLP for named entity recognition
- Fuzzy matching against skills dictionary
- Confidence scoring for skill relevance

### 2. Job Classification
- TF-IDF vectorization of job descriptions
- Multi-class classification into categories
- Feature importance analysis

### 3. User Matching
- Cosine similarity between job requirements and user skills
- Weighted scoring based on preferences
- Threshold-based filtering

## 📊 Dashboard Features

### Analytics View
- Job posting trends over time
- Skill demand analysis
- Salary distribution charts
- Company size breakdown

### Job Browser
- Real-time job search and filtering
- Skill highlighting
- Match score visualization
- Direct application links

### User Management
- Profile editing
- Notification preferences
- Match history tracking

## 🔧 API Endpoints (Future)

```python
# Planned REST API endpoints
GET /api/jobs              # List all jobs
GET /api/jobs/{id}         # Get specific job
POST /api/users            # Create user profile
PUT /api/users/{id}        # Update user preferences
GET /api/matches/{user_id} # Get user matches
```

## 📈 Performance Optimization

### Scraping
- Rotating user agents and proxies
- Rate limiting and backoff strategies
- Parallel processing with thread pools
- Caching mechanisms

### Machine Learning
- Batch processing for large datasets
- Model caching and versioning
- Incremental learning capabilities
- GPU acceleration support

### Database
- Indexing on frequently queried fields
- Data archiving for old jobs
- Connection pooling
- Query optimization

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_scraper.py
pytest tests/test_ml_categorizer.py

# Run with coverage
pytest --cov=src tests/
```

## 📝 Logging

Comprehensive logging system:

```python
# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
# Log files: logs/job_monitor.log, logs/scraper.log, logs/ml_models.log
# Rotation: Daily with 5 backup files
```

## 🚨 Error Handling

- Graceful degradation on scraping failures
- Retry mechanisms with exponential backoff
- Email notification for critical errors
- Health check endpoints

## 🔐 Security

- Environment variable protection
- Input validation and sanitization
- Rate limiting for API endpoints
- Secure email configurations

## 📦 Deployment

### Docker (Recommended)

```dockerfile
# Dockerfile example
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Cloud Deployment

- **AWS**: EC2 + RDS + SES
- **Google Cloud**: Compute Engine + Cloud SQL
- **Heroku**: Web dyno + Scheduler add-on

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Wiki](https://github.com/yourusername/job-monitoring-system/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/job-monitoring-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/job-monitoring-system/discussions)

## 🔄 Roadmap

### Version 2.0
- [ ] REST API implementation
- [ ] Multi-language support
- [ ] Advanced ML models (BERT, GPT)
- [ ] Mobile app companion

### Version 3.0
- [ ] Real-time job alerts
- [ ] Integration with job application platforms
- [ ] AI-powered resume matching
- [ ] Company culture analysis

## 📊 Statistics

- **100+ Technical Skills** in skills dictionary
- **8 Job Sites** supported (planned)
- **15+ Job Categories** for classification
- **50+ User Preferences** configurable
- **99%+ Uptime** target

---

**Made with ❤️ by the Job Monitoring Team**

*Happy job hunting! 🎯*