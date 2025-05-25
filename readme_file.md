# Job Monitoring and Categorization System

An automated system that monitors job postings from Karkidi.com daily and categorizes jobs based on required skills using unsupervised machine learning. The system notifies users when new jobs match their skill interests.

## Features

- **Daily Job Scraping**: Automatically scrapes job listings from Karkidi.com
- **Skill-Based Categorization**: Uses unsupervised clustering to classify jobs by required skills
- **User Matching**: Matches job categories with user skill profiles
- **Real-time Monitoring**: Continuously monitors for new job postings
- **Automated Notifications**: Alerts users when relevant jobs are found

## Project Structure

```
job-monitor-karkidi/
├── dashboard/              # Web dashboard for visualization
├── data/                   # Data storage and management
├── ml_categorizer/         # Machine learning models and training
├── monitoring/             # Job monitoring and user matching logic
├── preprocessing/          # Data cleaning and preprocessing
├── scraping/              # Web scraping modules
├── env_example.sh         # Environment variables template
├── readme_file            # Project documentation
└── requirements           # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/job-monitor-karkidi.git
cd job-monitor-karkidi
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements
```

4. Set up environment variables:
```bash
cp env_example.sh .env
# Edit .env file with your configuration
```

## Usage

### Data Collection

The scraping module collects job data from Karkidi.com including:
- Job titles
- Company names
- Required skills
- Job descriptions
- Location information
- Posting dates

### Machine Learning Pipeline

1. **Data Preprocessing**: Clean and prepare job data for analysis
2. **Feature Extraction**: Extract relevant features from job descriptions and skills
3. **Clustering**: Apply unsupervised learning to categorize jobs by skill requirements
4. **Model Training**: Train and save models for future job classification

### Monitoring System

The monitoring component:
- Runs daily to check for new job postings
- Applies trained models to categorize new jobs
- Matches job categories with user profiles
- Sends notifications for relevant matches

## Components

### Scraping Module
Located in the `scraping/` directory, handles web scraping operations to collect job data from target websites.

### Machine Learning Categorizer
The `ml_categorizer/` directory contains algorithms for:
- Unsupervised clustering of jobs based on skills
- Feature engineering for job descriptions
- Model persistence and loading

### Data Management
The `data/` directory organizes:
- Raw scraped job data
- Processed datasets
- Trained machine learning models

### Preprocessing
Data cleaning and preparation scripts in the `preprocessing/` directory handle:
- Text normalization
- Skill extraction
- Data validation

### Monitoring
Real-time job monitoring functionality in the `monitoring/` directory includes:
- Continuous job checking
- User-job matching algorithms
- Notification systems

### Dashboard
Web-based interface in the `dashboard/` directory for:
- Visualizing job categories
- Managing user profiles
- Viewing job match results

## Configuration

Edit the environment configuration file to set:
- Database connection strings
- API keys for notification services
- Scraping intervals and targets
- Machine learning model parameters

## Development

### Running the System

1. Start the job scraper:
```bash
python -m scraping.main
```

2. Train the categorization model:
```bash
python -m ml_categorizer.train
```

3. Start the monitoring service:
```bash
python -m monitoring.main
```

4. Launch the dashboard:
```bash
python -m dashboard.app
```

### Adding New Features

- **New Scrapers**: Add scraping modules in the `scraping/` directory
- **ML Models**: Extend categorization algorithms in `ml_categorizer/`
- **Notifications**: Implement new notification channels in `monitoring/`

## Data Flow

1. **Scraping**: Collect job data from target websites
2. **Preprocessing**: Clean and standardize job information
3. **Training**: Apply unsupervised learning to discover job categories
4. **Monitoring**: Check for new jobs and classify them
5. **Matching**: Compare job categories with user preferences
6. **Notification**: Alert users about relevant opportunities

