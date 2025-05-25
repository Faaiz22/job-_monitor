"""
Main Job Monitoring System
Orchestrates the entire job monitoring pipeline
"""

import json
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

from config.settings import PATHS, LOGGING_CONFIG
from src.scrapers.karkidi_scraper import KarkidiScraper
from src.scrapers.external_scraper import ExternalScraper
from src.preprocessing.preprocessor import JobPreprocessor
from src.ml.ml_categorizer import MLCategorizer
from src.matching.job_matcher import JobMatcher
from src.notifications.email_notifier import EmailNotifier


class JobMonitorSystem:
    """
    Main orchestrator for the job monitoring system
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.setup_components()
        
        # File paths
        self.raw_jobs_file = PATHS['data_dir'] / 'raw_jobs' / 'raw_jobs.json'
        self.processed_jobs_file = PATHS['data_dir'] / 'processed_jobs' / 'processed_jobs.csv'
        self.categorized_jobs_file = PATHS['data_dir'] / 'processed_jobs' / 'categorized_jobs.json'
        
        self.logger.info("JobMonitorSystem initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('job_monitor')
        logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
        
        if not logger.handlers:
            # File handler
            handler = logging.FileHandler(
                PATHS['logs_dir'] / 'job_monitor.log'
            )
            formatter = logging.Formatter(LOGGING_CONFIG['format'])
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def setup_components(self):
        """Initialize all system components"""
        try:
            self.karkidi_scraper = KarkidiScraper()
            self.external_scraper = ExternalScraper()
            self.preprocessor = JobPreprocessor()
            self.ml_categorizer = MLCategorizer()
            self.job_matcher = JobMatcher()
            self.email_notifier = EmailNotifier()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def load_existing_jobs(self) -> List[Dict]:
        """Load existing jobs from file"""
        if self.raw_jobs_file.exists():
            try:
                with open(self.raw_jobs_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading existing jobs: {str(e)}")
                return []
        return []
    
    def save_jobs_to_file(self, jobs: List[Dict], file_path: Path):
        """Save jobs to JSON file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(jobs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving jobs to file: {str(e)}")
            raise
    
    def scrape_new_jobs(self) -> int:
        """
        Scrape new jobs from all sources
        
        Returns:
            Number of new jobs scraped
        """
        self.logger.info("Starting job scraping process...")
        
        new_jobs_count = 0
        
        try:
            # Load existing jobs to avoid duplicates
            existing_jobs = self.load_existing_jobs()
            existing_urls = {job.get('url', '') for job in existing_jobs if job.get('url')}
            
            # 1. Scrape from Karkidi
            self.logger.info("Scraping jobs from Karkidi...")
            karkidi_jobs = self.karkidi_scraper.scrape_karkidi_jobs(max_pages=5)
            
            # Filter out duplicate URLs
            new_karkidi_jobs = [
                job for job in karkidi_jobs 
                if job.get('url') and job['url'] not in existing_urls
            ]
            
            self.logger.info(f"Found {len(new_karkidi_jobs)} new jobs from Karkidi")
            new_jobs_count += len(new_karkidi_jobs)
            
            # 2. Scrape external job descriptions if we have URLs
            if new_karkidi_jobs:
                self.logger.info("Enhancing job descriptions from external URLs...")
                enhanced_jobs = []
                
                for job in new_karkidi_jobs:
                    if job.get('url'):
                        # Try to get full description from the job URL
                        full_description = self.external_scraper.scrape_job_description(job['url'])
                        if full_description:
                            job['full_description'] = full_description
                            job['description'] = full_description[:1000]  # Keep first 1000 chars as summary
                    
                    enhanced_jobs.append(job)
                
                # Save new jobs
                all_jobs = existing_jobs + enhanced_jobs
                self.save_jobs_to_file(all_jobs, self.raw_jobs_file)
                
                self.logger.info(f"Successfully scraped {new_jobs_count} new jobs")
            
            return new_jobs_count
            
        except Exception as e:
            self.logger.error(f"Error in scraping process: {str(e)}")
            return 0
    
    def process_jobs(self) -> bool:
        """
        Process and clean raw job data
        
        Returns:
            True if processing successful, False otherwise
        """
        self.logger.info("Starting job processing...")
        
        try:
            # Check if we have raw jobs to process
            if not self.raw_jobs_file.exists():
                self.logger.warning("No raw jobs file found")
                return False
            
            # Process jobs
            success = self.preprocessor.process_jobs(
                input_file=str(self.raw_jobs_file),
                output_file=str(self.processed_jobs_file)
            )
            
            if success:
                self.logger.info("Job processing completed successfully")
                return True
            else:
                self.logger.error("Job processing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in job processing: {str(e)}")
            return False
    
    def categorize_jobs(self) -> bool:
        """
        Categorize jobs using ML model
        
        Returns:
            True if categorization successful, False otherwise
        """
        self.logger.info("Starting job categorization...")
        
        try:
            # Check if we have processed jobs
            if not self.processed_jobs_file.exists():
                self.logger.warning("No processed jobs file found")
                return False
            
            # Train or load model and categorize jobs
            categorized_jobs = self.ml_categorizer.categorize_jobs(str(self.processed_jobs_file))
            
            if categorized_jobs:
                # Save categorized jobs
                with open(self.categorized_jobs_file, 'w', encoding='utf-8') as f:
                    json.dump(categorized_jobs, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Successfully categorized {len(categorized_jobs)} jobs")
                return True
            else:
                self.logger.error("Job categorization failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in job categorization: {str(e)}")
            return False
    
    def match_and_notify_users(self) -> int:
        """
        Match jobs to users and send notifications
        
        Returns:
            Number of notifications sent
        """
        self.logger.info("Starting job matching and notification process...")
        
        notifications_sent = 0
        
        try:
            # Check if we have categorized jobs
            if not self.categorized_jobs_file.exists():
                self.logger.warning("No categorized jobs file found")
                return 0
            
            # Load categorized jobs
            with open(self.categorized_jobs_file, 'r', encoding='utf-8') as f:
                categorized_jobs = json.load(f)
            
            # Load user profiles
            user_profiles_file = PATHS['data_dir'] / 'user_profiles.json'
            if not user_profiles_file.exists():
                self.logger.warning("No user profiles file found")
                return 0
            
            with open(user_profiles_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            # Get active users
            active_users = [user for user in user_data.get('users', []) if user.get('active', True)]
            
            # Filter jobs from last 24 hours for daily notifications
            cutoff_time = datetime.now() - timedelta(days=1)
            recent_jobs = [
                job for job in categorized_jobs
                if self._is_recent_job(job, cutoff_time)
            ]
            
            self.logger.info(f"Found {len(recent_jobs)} recent jobs to match against {len(active_users)} active users")
            
            # Match jobs for each user
            for user in active_users:
                try:
                    # Find matching jobs for this user
                    matching_jobs = self.job_matcher.find_matching_jobs(user, recent_jobs)
                    
                    if matching_jobs:
                        # Send notification
                        success = self.email_notifier.send_job_notification(user, matching_jobs)
                        
                        if success:
                            notifications_sent += 1
                            self.logger.info(f"Sent notification to {user.get('name', 'Unknown')} with {len(matching_jobs)} job matches")
                        else:
                            self.logger.error(f"Failed to send notification to {user.get('name', 'Unknown')}")
                    else:
                        self.logger.debug(f"No matching jobs found for {user.get('name', 'Unknown')}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing user {user.get('name', 'Unknown')}: {str(e)}")
                    continue
            
            self.logger.info(f"Sent {notifications_sent} notifications")
            return notifications_sent
            
        except Exception as e:
            self.logger.error(f"Error in matching and notification process: {str(e)}")
            return 0
    
    def _is_recent_job(self, job: Dict, cutoff_time: datetime) -> bool:
        """Check if job is recent enough for notification"""
        try:
            scraped_date_str = job.get('scraped_date', '')
            if scraped_date_str:
                scraped_date = datetime.fromisoformat(scraped_date_str.replace('Z', '+00:00'))
                return scraped_date >= cutoff_time
        except:
            pass
        
        # If no valid date, assume it's recent
        return True
    
    def run_full_pipeline(self) -> Dict[str, int]:
        """
        Run the complete job monitoring pipeline
        
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting full job monitoring pipeline...")
        
        results = {
            'jobs_scraped': 0,
            'jobs_processed': 0,
            'jobs_categorized': 0,
            'notifications_sent': 0,
            'success': False
        }
        
        try:
            # Step 1: Scrape new jobs
            self.logger.info("Step 1: Scraping jobs...")
            results['jobs_scraped'] = self.scrape_new_jobs()
            
            if results['jobs_scraped'] == 0:
                self.logger.info("No new jobs found, checking existing jobs for processing...")
            
            # Step 2: Process jobs
            self.logger.info("Step 2: Processing jobs...")
            if self.process_jobs():
                # Count processed jobs
                if self.processed_jobs_file.exists():
                    df = pd.read_csv(self.processed_jobs_file)
                    results['jobs_processed'] = len(df)
            
            # Step 3: Categorize jobs
            self.logger.info("Step 3: Categorizing jobs...")
            if self.categorize_jobs():
                # Count categorized jobs
                if self.categorized_jobs_file.exists():
                    with open(self.categorized_jobs_file, 'r') as f:
                        categorized_jobs = json.load(f)
                        results['jobs_categorized'] = len(categorized_jobs)
            
            # Step 4: Match jobs and notify users
            self.logger.info("Step 4: Matching jobs and sending notifications...")
            results['notifications_sent'] = self.match_and_notify_users()
            
            # Mark as successful if we completed all steps
            results['success'] = True
            
            self.logger.info(f"Pipeline completed successfully: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in full pipeline: {str(e)}")
            results['success'] = False
            return results
    
    def schedule_monitoring(self):
        """Schedule the job monitoring to run daily"""
        self.logger.info("Setting up scheduled job monitoring...")
        
        # Schedule daily run at 9 AM
        schedule.every().day.at("09:00").do(self.run_full_pipeline)
        
        # Schedule a quick check every hour during business hours
        schedule.every().hour.do(lambda: self.scrape_new_jobs() if 9 <= datetime.now().hour <= 17 else None)
        
        self.logger.info("Scheduled monitoring set up successfully")
    
    def start_monitoring(self):
        """Start the continuous monitoring process"""
        self.logger.info("Starting continuous job monitoring...")
        
        # Set up scheduling
        self.schedule_monitoring()
        
        # Run initial pipeline
        self.logger.info("Running initial pipeline...")
        self.run_full_pipeline()
        
        # Keep the system running
        self.logger.info("Job monitoring system is now running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Job monitoring system stopped by user")
        except Exception as e:
            self.logger.error(f"Unexpected error in monitoring loop: {str(e)}")


def main():
    """Main function to run the job monitoring system"""
    try:
        # Initialize the system
        job_monitor = JobMonitorSystem()
        
        # Start monitoring
        job_monitor.start_monitoring()
        
    except Exception as e:
        logging.error(f"Failed to start job monitoring system: {str(e)}")
        raise


if __name__ == "__main__":
    main()
