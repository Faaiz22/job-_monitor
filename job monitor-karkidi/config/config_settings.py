"""
Configuration settings for the Job Monitoring System
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Directory paths
PATHS = {
    'base_dir': BASE_DIR,
    'data_dir': BASE_DIR / 'data',
    'logs_dir': BASE_DIR / 'logs',
    'models_dir': BASE_DIR / 'models',
    'config_dir': BASE_DIR / 'config',
    'src_dir': BASE_DIR / 'src',
}

# Create directories if they don't exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Scraping configuration
SCRAPING_CONFIG = {
    'karkidi_base_url': 'https://www.karkidi.com',
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'delay_between_requests': 2,  # seconds
    'max_retries': 3,
    'timeout': 30,
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
}

# Machine Learning configuration
ML_CONFIG = {
    'model_file': PATHS['models_dir'] / 'job_categorizer_model.pkl',
    'vectorizer_file': PATHS['models_dir'] / 'tfidf_vectorizer.pkl',
    'label_encoder_file': PATHS['models_dir'] / 'label_encoder.pkl',
    'scaler_file': PATHS['models_dir'] / 'feature_scaler.pkl',
    'n_clusters': 8,  # Number of job categories to discover
    'max_features': 5000,  # For TF-IDF vectorizer
    'min_df': 2,  # Minimum document frequency
    'max_df': 0.95,  # Maximum document frequency
    'random_state': 42,
}

# Email configuration
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'email_address': os.getenv('EMAIL_ADDRESS'),
    'email_password': os.getenv('EMAIL_PASSWORD'),
    'sender_name': 'Job Monitor System',
    'use_tls': True,
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# Job matching configuration
MATCHING_CONFIG = {
    'min_skill_match_score': 0.3,  # Minimum similarity score for skill matching
    'min_title_match_score': 0.2,  # Minimum similarity score for title matching
    'category_weight': 0.4,        # Weight for category matching
    'skill_weight': 0.4,           # Weight for skill matching
    'title_weight': 0.2,           # Weight for title matching
    'max_jobs_per_notification': 10,  # Maximum jobs to send in one notification
}

# Database configuration (if you want to add database support later)
DATABASE_CONFIG = {
    'type': 'sqlite',  # sqlite, postgresql, mysql
    'name': 'job_monitor.db',
    'path': PATHS['data_dir'] / 'job_monitor.db',
}

# Skills extraction patterns
SKILLS_PATTERNS = {
    'technical_skills': [
        r'\b(?:python|java|javascript|react|angular|vue|node\.?js|php|ruby|go|rust|swift|kotlin|scala|r|matlab)\b',
        r'\b(?:html|css|sass|scss|typescript|jquery|bootstrap|tailwind)\b',
        r'\b(?:sql|mysql|postgresql|mongodb|redis|elasticsearch|cassandra|oracle)\b',
        r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins|git|github|gitlab|bitbucket)\b',
        r'\b(?:linux|unix|windows|macos|ubuntu|centos|debian)\b',
        r'\b(?:apache|nginx|tomcat|iis|express|flask|django|spring|laravel)\b',
        r'\b(?:machine learning|artificial intelligence|deep learning|tensorflow|pytorch|scikit-learn)\b',
        r'\b(?:data science|data analysis|pandas|numpy|matplotlib|seaborn|plotly)\b',
    ],
    'soft_skills': [
        r'\b(?:communication|teamwork|leadership|problem solving|analytical|creative)\b',
        r'\b(?:project management|agile|scrum|kanban|waterfall)\b',
        r'\b(?:presentation|negotiation|customer service|sales)\b',
    ],
    'languages': [
        r'\b(?:english|french|spanish|german|italian|portuguese|russian|chinese|japanese|korean|arabic)\b',
    ]
}

# Job categories mapping
JOB_CATEGORIES = {
    0: 'Software Development',
    1: 'Data Science & Analytics', 
    2: 'DevOps & Cloud',
    3: 'Web Development',
    4: 'Mobile Development',
    5: 'UI/UX Design',
    6: 'Project Management',
    7: 'IT Support & Administration',
    8: 'Quality Assurance',
    9: 'Cybersecurity',
    10: 'Database Administration',
    11: 'Business Intelligence',
    12: 'Marketing & Sales',
    13: 'HR & Recruitment',
    14: 'Finance & Accounting',
    15: 'Other'
}

# Validation settings
VALIDATION_CONFIG = {
    'required_fields': ['title', 'company', 'location'],
    'min_description_length': 50,
    'max_title_length': 200,
    'max_company_length': 100,
}
