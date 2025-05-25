"""
Main Pipeline Orchestrator for Job Monitoring System.
Coordinates all modules: scraping, preprocessing, ML clustering, matching, and notifications.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from scrapers.job_scraper import JobScraper
from preprocessing.data_processor import DataProcessor
from ml_categorizer.clustering_model import JobClusterer
from matching.job_matcher import JobMatcher
from notifications.alert_system import AlertSystem

class JobMonitoringPipeline:
    """
    Main orchestrator for the job monitoring system.
    Handles initial training and daily update modes.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.setup_logging()
        self.setup_directories()
        
        # Initialize components
        self.scraper = JobScraper()
        self.processor = DataProcessor()
        self.clusterer = JobClusterer()
        self.matcher = JobMatcher()
        self.alert_system = AlertSystem()
        
        # File paths
        self.raw_jobs_path = "data/raw_jobs.csv"
        self.processed_jobs_path = "data/processed_jobs.csv"
        self.user_preferences_path = "config/user_preferences.yaml"
        
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Setup logging
        log_filename = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            "data", "models", "config", "logs", 
            "notifications", "reports"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def check_existing_models(self) -> dict:
        """
        Check which models/data already exist.
        
        Returns:
            Dictionary with existence status of key files
        """
        files_to_check = {
            'raw_data': self.raw_jobs_path,
            'processed_data': self.processed_jobs_path,
            'tfidf_vectorizer': "models/tfidf_vectorizer.pkl",
            'cluster_model': "models/job_cluster_model.pkl",
            'user_preferences': self.user_preferences_path
        }
        
        status = {}
        for key, path in files_to_check.items():
            status[key] = os.path.exists(path)
            self.logger.info(f"{key}: {'EXISTS' if status[key] else 'NOT FOUND'} - {path}")
        
        return status
    
    def run_initial_training(self, job_sites: list = None, max_jobs: int = 1000):
        """
        Run initial training pipeline - scrapes data and trains all models.
        
        Args:
            job_sites: List of job sites to scrape
            max_jobs: Maximum number of jobs to scrape
        """
        self.logger.info("=== STARTING INITIAL TRAINING PIPELINE ===")
        
        try:
            # Step 1: Scrape jobs
            self.logger.info("Step 1: Scraping job data...")
            job_sites = job_sites or ['indeed', 'glassdoor']
            
            all_jobs = []
            for site in job_sites:
                try:
                    jobs = self.scraper.scrape_jobs(
                        site=site,
                        search_terms=["python", "data science", "software engineer"],
                        locations=["New York", "San Francisco", "Remote"],
                        max_jobs=max_jobs // len(job_sites)
                    )
                    if jobs:
                        all_jobs.extend(jobs)
                        self.logger.info(f"Scraped {len(jobs)} jobs from {site}")
                except Exception as e:
                    self.logger.error(f"Error scraping {site}: {e}")
            
            if not all_jobs:
                raise Exception("No jobs scraped from any site")
            
            # Save raw data
            df_raw = pd.DataFrame(all_jobs)
            df_raw.to_csv(self.raw_jobs_path, index=False)
            self.logger.info(f"Saved {len(all_jobs)} raw jobs to {self.raw_jobs_path}")
            
            # Step 2: Process data
            self.logger.info("Step 2: Processing job data...")
            df_processed = self.processor.process_jobs(df_raw)
            
            if df_processed is None or df_processed.empty:
                raise Exception("Data processing failed")
            
            df_processed.to_csv(self.processed_jobs_path, index=False)
            self.logger.info(f"Processed data saved to {self.processed_jobs_path}")
            
            # Step 3: Train clustering model
            self.logger.info("Step 3: Training clustering model...")
            df_clustered, cluster_labels = self.clusterer.train_model(df_processed)
            self.clusterer.save_model()
            
            # Update processed data with clusters
            df_clustered.to_csv(self.processed_jobs_path, index=False)
            self.logger.info("Clustering model trained and saved")
            
            # Step 4: Generate initial matches (if preferences exist)
            if os.path.exists(self.user_preferences_path):
                self.logger.info("Step 4: Generating initial matches...")
                matches = self.matcher.find_matches(df_clustered)
                if matches:
                    self.logger.info(f"Found {len(matches)} initial matches")
            
            self.logger.info("=== INITIAL TRAINING COMPLETED SUCCESSFULLY ===")
            
        except Exception as e:
            self.logger.error(f"Initial training failed: {e}")
            raise
    
    def run_daily_update(self, job_sites: list = None, max_jobs: int = 200):
        """
        Run daily update pipeline - scrapes new data and updates models.
        
        Args:
            job_sites: List of job sites to scrape
            max_jobs: Maximum number of jobs to scrape
        """
        self.logger.info("=== STARTING DAILY UPDATE PIPELINE ===")
        
        try:
            # Check if models exist
            status = self.check_existing_models()
            if not status['cluster_model'] or not status['tfidf_vectorizer']:
                self.logger.warning("Required models not found. Running initial training...")
                self.run_initial_training(job_sites, max_jobs)
                return
            
            # Load existing models
            self.logger.info("Loading existing models...")
            if not self.clusterer.load_model():
                raise Exception("Failed to load clustering model")
            
            # Load existing data
            df_existing = pd.DataFrame()
            if status['processed_data']:
                df_existing = pd.read_csv(self.processed_jobs_path)
                self.logger.info(f"Loaded {len(df_existing)} existing jobs")
            
            # Step 1: Scrape new jobs
            self.logger.info("Step 1: Scraping new job data...")
            job_sites = job_sites or ['indeed', 'glassdoor']
            
            all_new_jobs = []
            for site in job_sites:
                try:
                    jobs = self.scraper.scrape_jobs(
                        site=site,
                        search_terms=["python", "data science", "software engineer"],
                        locations=["New York", "San Francisco", "Remote"],
                        max_jobs=max_jobs // len(job_sites)
                    )
                    if jobs:
                        all_new_jobs.extend(jobs)
                except Exception as e:
                    self.logger.error(f"Error scraping {site}: {e}")
            
            if not all_new_jobs:
                self.logger.info("No new jobs found")
                return
            
            # Step 2: Process new data
            self.logger.info("Step 2: Processing new job data...")
            df_new_raw = pd.DataFrame(all_new_jobs)
            df_new_processed = self.processor.process_jobs(df_new_raw)
            
            if df_new_processed is None or df_new_processed.empty:
                self.logger.warning("No new jobs after processing")
                return
            
            # Step 3: Add clusters to new data
            self.logger.info("Step 3: Categorizing new jobs...")
            df_new_clustered = self.clusterer.add_clusters_to_df(df_new_processed)
            
            # Step 4: Combine with existing data (remove duplicates)
            if not df_existing.empty:
                # Simple duplicate detection based on title and company
                existing_jobs = set(
                    zip(df_existing['title'].fillna(''), df_existing['company'].fillna(''))
                )
                
                new_jobs_mask = ~df_new_clustered.apply(
                    lambda row: (row['title'] or '', row['company'] or '') in existing_jobs,
                    axis=1
                )
                
                df_truly_new = df_new_clustered[new_jobs_mask]
                self.logger.info(f"Found {len(df_truly_new)} truly new jobs after deduplication")
                
                if not df_truly_new.empty:
                    df_combined = pd.concat([df_existing, df_truly_new], ignore_index=True)
                else:
                    df_combined = df_existing
            else:
                df_combined = df_new_clustered
                df_truly_new = df_new_clustered
            
            # Save updated data
            df_combined.to_csv(self.processed_jobs_path, index=False)
            
            # Step 5: Find matches and send notifications
            if not df_truly_new.empty and os.path.exists(self.user_preferences_path):
                self.logger.info("Step 5: Finding matches and sending notifications...")
                matches = self.matcher.find_matches(df_truly_new)
                
                if matches:
                    self.logger.info(f"Found {len(matches)} new matches")
                    
                    # Send notifications
                    notification_sent = self.alert_system.send_job_alerts(matches)
                    if notification_sent:
                        self.logger.info("Notifications sent successfully")
                    else:
                        self.logger.warning("Failed to send notifications")
                else:
                    self.logger.info("No matches found for user preferences")
            
            self.logger.info("=== DAILY UPDATE COMPLETED SUCCESSFULLY ===")
            
        except Exception as e:
            self.logger.error(f"Daily update failed: {e}")
            raise
    
    def run_analysis_only(self):
        """
        Run analysis on existing data without scraping new data.
        """
        self.logger.info("=== RUNNING ANALYSIS ON EXISTING DATA ===")
        
        try:
            status = self.check_existing_models()
            
            if not status['processed_data']:
                self.logger.error("No processed data found. Run initial training first.")
                return
            
            # Load data
            df = pd.read_csv(self.processed_jobs_path)
            self.logger.info(f"Loaded {len(df)} jobs for analysis")
            
            # Load or train clustering model
            if status['cluster_model']:
                self.clusterer.load_model()
                self.logger.info("Loaded existing clustering model")
            else:
                self.logger.info("Training new clustering model...")
                df, cluster_labels = self.clusterer.train_model(df)
                self.clusterer.save_model()
                df.to_csv(self.processed_jobs_path, index=False)
            
            # Generate matches if preferences exist
            if status['user_preferences']:
                matches = self.matcher.find_matches(df)
                if matches:
                    self.logger.info(f"Found {len(matches)} matches")
                    
                    # Generate analysis report
                    self.generate_analysis_report(df, matches)
                else:
                    self.logger.info("No matches found")
            
            self.logger.info("=== ANALYSIS COMPLETED ===")
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    def generate_analysis_report(self, df: pd.DataFrame, matches: list = None):
        """
        Generate analysis report.
        
        Args:
            df: Job data DataFrame
            matches: List of matched jobs
        """
        report_path = f"reports/analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("JOB MONITORING SYSTEM - ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic statistics
            f.write("BASIC STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Jobs: {len(df)}\n")
            f.write(f"Unique Companies: {df['company'].nunique()}\n")
            f.write(f"Unique Locations: {df['location'].nunique()}\n\n")
            
            # Cluster analysis
            if 'cluster' in df.columns:
                f.write("CLUSTER ANALYSIS\n")
                f.write("-" * 20 + "\n")
                cluster_counts = df['cluster'].value_counts().sort_index()
                for cluster_id, count in cluster_counts.items():
                    cluster_label = self.clusterer.cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
                    f.write(f"Cluster {cluster_id}: {count} jobs - {cluster_label}\n")
                f.write("\n")
            
            # Top companies
            f.write("TOP COMPANIES\n")
            f.write("-" * 20 + "\n")
            top_companies = df['company'].value_counts().head(10)
            for company, count in top_companies.items():
                f.write(f"{company}: {count} jobs\n")
            f.write("\n")
            
            # Matches summary
            if matches:
                f.write("MATCHES SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Matches: {len(matches)}\n")
                
                # Top matching companies
                match_companies = [job.get('company', 'Unknown') for job in matches]
                match_company_counts = pd.Series(match_companies).value_counts().head(5)
                f.write("Top Matching Companies:\n")
                for company, count in match_company_counts.items():
                    f.write(f"  {company}: {count} matches\n")
        
        self.logger.info(f"Analysis report saved to {report_path}")


def main():
    """Main function to run the pipeline with command line arguments."""
    parser = argparse.ArgumentParser(description="Job Monitoring System Pipeline")
    parser.add_argument(
        'mode',
        choices=['initial', 'daily', 'analysis'],
        help='Pipeline mode: initial (full training), daily (update), or analysis (existing data)'
    )
    parser.add_argument(
        '--sites',
        nargs='+',
        default=['indeed'],
        help='Job sites to scrape (default: indeed)'
    )
    parser.add_argument(
        '--max-jobs',
        type=int,
        default=1000,
        help='Maximum jobs to scrape (default: 1000 for initial, 200 for daily)'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = JobMonitoringPipeline(args.config)
    
    try:
        if args.mode == 'initial':
            max_jobs = args.max_jobs if args.max_jobs != 1000 else 1000
            pipeline.run_initial_training(args.sites, max_jobs)
        elif args.mode == 'daily':
            max_jobs = args.max_jobs if args.max_jobs != 1000 else 200
            pipeline.run_daily_update(args.sites, max_jobs)
        elif args.mode == 'analysis':
            pipeline.run_analysis_only()
        
        print(f"\n✅ Pipeline completed successfully in '{args.mode}' mode!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
        