"""
Configuration settings for the Job Monitoring System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# ===========================================
# JOB SITES CONFIGURATION
# ===========================================

JOB_SITES = {
    'indeed': {
        'base_url': 'https://www.indeed.com',
        'search_url': 'https://www.indeed.com/jobs',
        'enabled': True,
        'max_pages': int(os.getenv('MAX_PAGES_PER_SITE', 10)),
        'selectors': {
            'job_cards': '[data-jk]',
            'title': 'h2 a span',
            'company': '.companyName',
            'location': '.companyLocation',
            'description': '.job-snippet',
            'date': '.date',
            'salary': '.salary-snippet'
        }
    },
    'linkedin': {
        'base_url': 'https://www.linkedin.com',
        'search_url': 'https://www.linkedin.com/jobs/search',
        'enabled': True,
        'max_pages': int(os.getenv('MAX_PAGES_PER_SITE', 10)),
        'selectors': {
            'job_cards': '.job-search-card',
            'title': '.job-search-card__title',
            'company': '.job-search-card__subtitle-primary',
            'location': '.job-search-card__subtitle-secondary',
            'description': '.job-search-card__snippet',
            'date': '.job-search-card__listdate',
            'link': '.job-search-card__title-link'
        }
    },
    'glassdoor': {
        'base_url': 'https://www.glassdoor.com',
        'search_url': 'https://www.glassdoor.com/Job/jobs.htm',
        'enabled': True,
        'max_pages': int(os.getenv('MAX_PAGES_PER_SITE', 10)),
        'selectors': {
            'job_cards': '.react-job-listing',
            'title': '[data-test="job-title"]',
            'company': '[data-test="employer-name"]',
            'location': '[data-test="job-location"]',
            'description': '[data-test="job-description"]',
            'salary': '[data-test="detailSalary"]'
        }
    }
}

# ===========================================
# SEARCH PARAMETERS
# ===========================================

SEARCH_KEYWORDS = [
    'software developer',
    'python developer',
    'data scientist',
    'machine learning engineer',
    'full stack developer',
    'backend developer',
    'frontend developer',
    'devops engineer',
    'data analyst',
    'software engineer'
]

SEARCH_LOCATIONS = [
    'Remote',
    'New York, NY',
    'San Francisco, CA',
    'Seattle, WA',
    'Austin, TX',
    'Chicago, IL',
    'Boston, MA',
    'Los Angeles, CA'
]

# ===========================================
# SCRAPING CONFIGURATION
# ===========================================

SCRAPER_CONFIG = {
    'delay_min': int(os.getenv('SCRAPER_DELAY_MIN', 2)),
    'delay_max': int(os.getenv('SCRAPER_DELAY_MAX', 5)),
    'timeout': int(os.getenv('SCRAPER_TIMEOUT', 30)),
    'max_retries': 3,
    'user_agent': os.getenv('USER_AGENT', 
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'),
    'headers': {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
}

# Selenium WebDriver configuration
WEBDRIVER_CONFIG = {
    'path': os.getenv('WEBDRIVER_PATH', './drivers/chromedriver'),
    'headless': os.getenv('WEBDRIVER_HEADLESS', 'True').lower() == 'true',
    'window_size': os.getenv('WEBDRIVER_WINDOW_SIZE', '1920,1080'),
    'options': [
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--disable-extensions',
        '--disable-plugins',
        '--disable-images'
    ]
}

# ===========================================
# DATABASE CONFIGURATION
# ===========================================

DATABASE_CONFIG = {
    'url': os.getenv('DATABASE_URL', 'sqlite:///job_monitoring.db'),
    'path': os.getenv('DATABASE_PATH', str(BASE_DIR / 'data' / 'job_monitoring.db')),
    'backup_frequency': 'daily',
    'retention_days': 30
}

# ===========================================
# FILE PATHS
# ===========================================

PATHS = {
    'data_dir': BASE_DIR / 'data',
    'models_dir': BASE_DIR / 'models',
    'logs_dir': BASE_DIR / 'logs',
    'config_dir': BASE_DIR / 'config',
    
    # Data files
    'skills_dictionary': BASE_DIR / 'data' / 'skills_dictionary.json',
    'user_profiles': BASE_DIR / 'data' / 'user_profiles.json',
    'raw_jobs': BASE_DIR / 'data' / 'raw',
    'processed_jobs': BASE_DIR / 'data' / 'processed',
    
    # Model files
    'trained_models': BASE_DIR / 'models' / 'trained_models',
    'model_artifacts': BASE_DIR / 'models' / 'model_artifacts',
    
    # Log files
    'main_log': BASE_DIR / 'logs' / 'job_monitor.log',
    'scraper_log': BASE_DIR / 'logs' / 'scraper.log',
    'ml_log': BASE_DIR / 'logs' / 'ml_models.log'
}

# ===========================================
# EMAIL CONFIGURATION
# ===========================================

EMAIL_CONFIG = {
    'host': os.getenv('EMAIL_HOST', 'smtp.gmail.com'),
    'port': int(os.getenv('EMAIL_PORT', 587)),
    'use_tls': os.getenv('EMAIL_USE_TLS', 'True').lower() == 'true',
    'username': os.getenv('EMAIL_HOST_USER'),
    'password': os.getenv('EMAIL_HOST_PASSWORD'),
    'from_name': os.getenv('EMAIL_FROM_NAME', 'Job Monitor System')
}

NOTIFICATION_CONFIG = {
    'frequency': os.getenv('NOTIFICATION_FREQUENCY', 'daily'),
    'max_jobs_per_email': int(os.getenv('MAX_JOBS_PER_EMAIL', 10)),
    'notification_time': os.getenv('NOTIFICATION_TIME', '09:00'),
    'template_path': BASE_DIR / 'src' / 'matching_notifier' / 'templates'
}

# ===========================================
# MACHINE LEARNING CONFIGURATION
# ===========================================

ML_CONFIG = {
    'spacy_model': os.getenv('SPACY_MODEL', 'en_core_web_sm'),
    'min_confidence': float(os.getenv('MIN_CONFIDENCE_THRESHOLD', 0.7)),
    'model_path': os.getenv('ML_MODEL_PATH', str(BASE_DIR / 'models' / 'trained_models')),
    'feature_extraction': {
        'max_features': 5000,
        'min_df': 2,
        'max_df': 0.95,
        'stop_words': 'english'
    },
    'classification': {
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5
    }
}

# Job categories for classification
JOB_CATEGORIES = [
    'Software Development',
    'Data Science',
    'Machine Learning',
    'DevOps',
    'Frontend Development',
    'Backend Development',
    'Full Stack Development',
    'Mobile Development',
    'QA/Testing',
    'Product Management',
    'UI/UX Design',
    'Data Engineering',
    'Cybersecurity',
    'Cloud Engineering',
    'Other'
]

# ===========================================
# DASHBOARD CONFIGURATION
# ===========================================

DASHBOARD_CONFIG = {
    'title': os.getenv('DASHBOARD_TITLE', 'Job Monitoring Dashboard'),
    'port': int(os.getenv('STREAMLIT_PORT', 8501)),
    'theme': os.getenv('DASHBOARD_THEME', 'light'),
    'page_config': {
        'page_title': 'Job Monitor',
        'page_icon': '💼',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    },
    'charts_config': {
        'color_scheme': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'default_height': 400
    }
}

# ===========================================
# RATE LIMITING & PERFORMANCE
# ===========================================

RATE_LIMITING = {
    'requests_per_minute': int(os.getenv('REQUESTS_PER_MINUTE', 30)),
    'concurrent_requests': int(os.getenv('CONCURRENT_REQUESTS', 5)),
    'backoff_factor': 1.5,
    'max_backoff_time': 300
}

CACHE_CONFIG = {
    'expire_hours': int(os.getenv('CACHE_EXPIRE_HOURS', 24)),
    'enabled': os.getenv('ENABLE_CACHING', 'True').lower() == 'true',
    'max_size': 1000
}

# ===========================================
# LOGGING CONFIGURATION
# ===========================================

LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': os.getenv('LOG_FILE', str(BASE_DIR / 'logs' / 'job_monitor.log')),
    'max_size': os.getenv('LOG_MAX_SIZE', '10MB'),
    'backup_count': int(os.getenv('LOG_BACKUP_COUNT', 5)),
    'rotation': 'daily'
}

# ===========================================
# DEVELOPMENT SETTINGS
# ===========================================

DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
TESTING = os.getenv('TESTING', 'False').lower() == 'true'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')

# Create directories if they don't exist
for path in PATHS.values():
    if isinstance(path, Path) and not path.name.endswith('.json') and not path.name.endswith('.log'):
        path.mkdir(parents=True, exist_ok=True)