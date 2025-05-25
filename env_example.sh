# ===========================================
# JOB MONITORING SYSTEM - ENVIRONMENT VARIABLES
# ===========================================
# Copy this file to .env and fill in your actual values

# Email Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
EMAIL_FROM_NAME=Job Monitor System

# Database Configuration
DATABASE_URL=sqlite:///job_monitoring.db
DATABASE_PATH=./data/job_monitoring.db

# Scraping Configuration
SCRAPER_DELAY_MIN=2
SCRAPER_DELAY_MAX=5
SCRAPER_TIMEOUT=30
MAX_PAGES_PER_SITE=10
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36

# LinkedIn API (if available)
LINKEDIN_CLIENT_ID=your-linkedin-client-id
LINKEDIN_CLIENT_SECRET=your-linkedin-client-secret
LINKEDIN_ACCESS_TOKEN=your-linkedin-access-token

# Indeed API (if available)
INDEED_PUBLISHER_ID=your-indeed-publisher-id

# Glassdoor API (if available)
GLASSDOOR_PARTNER_ID=your-glassdoor-partner-id
GLASSDOOR_KEY=your-glassdoor-key

# Machine Learning Configuration
ML_MODEL_PATH=./models/trained_models/
SPACY_MODEL=en_core_web_sm
MIN_CONFIDENCE_THRESHOLD=0.7

# Notification Settings
NOTIFICATION_FREQUENCY=daily
MAX_JOBS_PER_EMAIL=10
NOTIFICATION_TIME=09:00

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/job_monitor.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5

# Dashboard Configuration
STREAMLIT_PORT=8501
DASHBOARD_TITLE=Job Monitoring Dashboard
DASHBOARD_THEME=light

# File Paths
DATA_DIR=./data/
MODELS_DIR=./models/
LOGS_DIR=./logs/
CONFIG_DIR=./config/

# Rate Limiting
REQUESTS_PER_MINUTE=30
CONCURRENT_REQUESTS=5

# Development Settings
DEBUG=False
TESTING=False
ENVIRONMENT=production

# Selenium WebDriver
WEBDRIVER_PATH=./drivers/chromedriver
WEBDRIVER_HEADLESS=True
WEBDRIVER_WINDOW_SIZE=1920,1080

# Cache Settings
CACHE_EXPIRE_HOURS=24
ENABLE_CACHING=True

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here