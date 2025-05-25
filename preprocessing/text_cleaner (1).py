"""
Text cleaning utilities for job monitoring system.
Handles HTML removal, URL cleaning, and text normalization.
"""

import re
import html
import pandas as pd
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Handles text cleaning operations for job descriptions and content.
    """
    
    def __init__(self):
        """Initialize the TextCleaner with compiled regex patterns for efficiency."""
        # Compile regex patterns for better performance
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s\-.,!?()]')
        
    def remove_html(self, text: str) -> str:
        """
        Remove HTML tags and decode HTML entities.
        
        Args:
            text (str): Text containing HTML
            
        Returns:
            str: Clean text without HTML
        """
        if not isinstance(text, str):
            return ""
            
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text
    
    def remove_urls_and_contacts(self, text: str) -> str:
        """
        Remove URLs, email addresses, and phone numbers.
        
        Args:
            text (str): Text containing URLs and contact info
            
        Returns:
            str: Clean text without URLs and contact info
        """
        if not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' ', text)
        
        # Remove phone numbers
        text = self.phone_pattern.sub(' ', text)
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by handling whitespace and special characters.
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove excessive special characters (keep basic punctuation)
        text = self.special_chars_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Perform complete text cleaning pipeline.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Fully cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
            
        try:
            # Apply cleaning pipeline
            text = self.remove_html(text)
            text = self.remove_urls_and_contacts(text)
            text = self.normalize_text(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {e}")
            return ""
    
    def clean_job_data(self, df: pd.DataFrame, 
                      text_columns: List[str] = None) -> pd.DataFrame:
        """
        Clean text columns in a job dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame containing job data
            text_columns (List[str]): Columns to clean. If None, defaults to common job columns
            
        Returns:
            pd.DataFrame: DataFrame with cleaned text columns
        """
        if text_columns is None:
            text_columns = ['description', 'title', 'company', 'requirements', 'summary']
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Clean each specified text column
        for column in text_columns:
            if column in df_clean.columns:
                logger.info(f"Cleaning column: {column}")
                df_clean[f'cleaned_{column}'] = df_clean[column].apply(self.clean_text)
            else:
                logger.warning(f"Column '{column}' not found in DataFrame")
        
        return df_clean
    
    def create_combined_text(self, df: pd.DataFrame, 
                           columns: List[str] = None,
                           output_column: str = 'combined_text') -> pd.DataFrame:
        """
        Combine multiple cleaned text columns into a single column.
        
        Args:
            df (pd.DataFrame): DataFrame with cleaned text columns
            columns (List[str]): Columns to combine. If None, uses default job columns
            output_column (str): Name for the combined text column
            
        Returns:
            pd.DataFrame: DataFrame with combined text column
        """
        if columns is None:
            columns = ['cleaned_title', 'cleaned_description', 'cleaned_requirements']
        
        df_combined = df.copy()
        
        # Combine available columns
        available_columns = [col for col in columns if col in df_combined.columns]
        
        if not available_columns:
            logger.warning("No specified columns found for combining")
            df_combined[output_column] = ""
            return df_combined
        
        logger.info(f"Combining columns: {available_columns}")
        
        # Combine text with space separation
        df_combined[output_column] = df_combined[available_columns].apply(
            lambda row: ' '.join(str(val) for val in row if pd.notna(val) and str(val).strip()),
            axis=1
        )
        
        return df_combined
    
    def process_job_file(self, input_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Process a job data file with complete text cleaning pipeline.
        
        Args:
            input_path (str): Path to input job data file
            output_path (str): Path to save processed data. If None, doesn't save
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        try:
            logger.info(f"Loading job data from: {input_path}")
            
            # Load data (handle both JSON and CSV)
            if input_path.endswith('.json'):
                df = pd.read_json(input_path)
            elif input_path.endswith('.csv'):
                df = pd.read_csv(input_path)
            else:
                raise ValueError("Unsupported file format. Use .json or .csv")
            
            logger.info(f"Loaded {len(df)} job records")
            
            # Clean text columns
            df_clean = self.clean_job_data(df)
            
            # Create combined text column
            df_clean = self.create_combined_text(df_clean)
            
            # Save if output path specified
            if output_path:
                logger.info(f"Saving cleaned data to: {output_path}")
                if output_path.endswith('.csv'):
                    df_clean.to_csv(output_path, index=False)
                elif output_path.endswith('.json'):
                    df_clean.to_json(output_path, orient='records', indent=2)
                else:
                    # Default to CSV
                    df_clean.to_csv(output_path, index=False)
            
            logger.info("Text cleaning completed successfully")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error processing job file: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    cleaner = TextCleaner()
    
    # Process job data file
    try:
        df = cleaner.process_job_file(
            input_path="data/raw_jobs/raw_jobs.json",
            output_path="data/processed_jobs/cleaned_jobs.csv"
        )
        print(f"Processed {len(df)} job records")
        print("\nCleaned columns:")
        print([col for col in df.columns if col.startswith('cleaned_')])
        
    except FileNotFoundError:
        print("Job data file not found. Please ensure raw job data exists.")
    except Exception as e:
        print(f"Error: {e}")
