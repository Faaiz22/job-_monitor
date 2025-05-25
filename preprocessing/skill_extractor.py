"""
Skill extraction utilities for job monitoring system.
Uses spaCy and skills dictionary to extract relevant skills from job descriptions.
"""

import json
import re
import pandas as pd
import spacy
from typing import List, Dict, Set, Any
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillExtractor:
    """
    Extracts technical and soft skills from job descriptions using NLP and pattern matching.
    """
    
    def __init__(self, skills_dict_path: str = "data/skills_dictionary.json"):
        """
        Initialize the SkillExtractor with skills dictionary and spaCy model.
        
        Args:
            skills_dict_path (str): Path to skills dictionary JSON file
        """
        self.skills_dict_path = skills_dict_path
        self.skills_dict = self._load_skills_dictionary()
        self.nlp = self._load_spacy_model()
        
        # Create lookup sets for efficient matching
        self._create_skill_lookups()
        
    def _load_skills_dictionary(self) -> Dict[str, Any]:
        """Load skills dictionary from JSON file."""
        try:
            with open(self.skills_dict_path, 'r', encoding='utf-8') as f:
                skills_dict = json.load(f)
            logger.info(f"Loaded skills dictionary from {self.skills_dict_path}")
            return skills_dict
        except FileNotFoundError:
            logger.error(f"Skills dictionary not found at {self.skills_dict_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing skills dictionary: {e}")
            raise
    
    def _load_spacy_model(self):
        """Load spaCy model for NLP processing."""
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy en_core_web_sm model")
            return nlp
        except OSError:
            logger.warning("spaCy en_core_web_sm model not found. Install with: python -m spacy download en_core_web_sm")
            logger.info("Falling back to simple pattern matching")
            return None
    
    def _create_skill_lookups(self):
        """Create efficient lookup dictionaries for skill matching."""
        self.all_skills = set()
        self.skill_patterns = {}
        
        # Process technical skills
        tech_skills = self.skills_dict.get('technical_skills', {})
        for category, skills in tech_skills.items():
            for skill in skills:
                self.all_skills.add(skill.lower())
                # Create regex pattern for exact and partial matches
                pattern = re.compile(rf'\b{re.escape(skill.lower())}\b', re.IGNORECASE)
                self.skill_patterns[skill.lower()] = pattern
        
        # Process soft skills
        soft_skills = self.skills_dict.get('soft_skills', {})
        for category, skills in soft_skills.items():
            for skill in skills:
                self.all_skills.add(skill.lower())
                pattern = re.compile(rf'\b{re.escape(skill.lower())}\b', re.IGNORECASE)
                self.skill_patterns[skill.lower()] = pattern
        
        # Add domain knowledge
        domains = self.skills_dict.get('industry_domains', [])
        for domain in domains:
            self.all_skills.add(domain.lower())
            pattern = re.compile(rf'\b{re.escape(domain.lower())}\b', re.IGNORECASE)
            self.skill_patterns[domain.lower()] = pattern
        
        # Add certifications
        certs = self.skills_dict.get('certifications', [])
        for cert in certs:
            self.all_skills.add(cert.lower())
            pattern = re.compile(rf'\b{re.escape(cert.lower())}\b', re.IGNORECASE)
            self.skill_patterns[cert.lower()] = pattern
        
        logger.info(f"Created lookup patterns for {len(self.all_skills)} skills")
    
    def extract_skills_pattern_matching(self, text: str) -> List[str]:
        """
        Extract skills using pattern matching (fallback method).
        
        Args:
            text (str): Text to extract skills from
            
        Returns:
            List[str]: List of extracted skills
        """
        if not text or not isinstance(text, str):
            return []
        
        found_skills = set()
        text_lower = text.lower()
        
        # Use compiled regex patterns for efficient matching
        for skill, pattern in self.skill_patterns.items():
            if pattern.search(text_lower):
                found_skills.add(skill)
        
        return list(found_skills)
    
    def extract_skills_spacy(self, text: str) -> List[str]:
        """
        Extract skills using spaCy NLP processing.
        
        Args:
            text (str): Text to extract skills from
            
        Returns:
            List[str]: List of extracted skills
        """
        if not text or not isinstance(text, str) or not self.nlp:
            return self.extract_skills_pattern_matching(text)
        
        try:
            doc = self.nlp(text)
            found_skills = set()
            
            # Extract skills from tokens and entities
            for token in doc:
                token_text = token.text.lower()
                if token_text in self.all_skills:
                    found_skills.add(token_text)
            
            # Also check multi-word skills
            text_lower = text.lower()
            for skill in self.all_skills:
                if len(skill.split()) > 1:  # Multi-word skills
                    if skill in text_lower:
                        found_skills.add(skill)
            
            return list(found_skills)
            
        except Exception as e:
            logger.warning(f"Error with spaCy processing, falling back to pattern matching: {e}")
            return self.extract_skills_pattern_matching(text)
    
    def extract_skills(self, text: str, method: str = "auto") -> List[str]:
        """
        Extract skills from text using specified method.
        
        Args:
            text (str): Text to extract skills from
            method (str): Extraction method - "auto", "spacy", or "pattern"
            
        Returns:
            List[str]: List of extracted skills
        """
        if method == "spacy" or (method == "auto" and self.nlp):
            return self.extract_skills_spacy(text)
        else:
            return self.extract_skills_pattern_matching(text)
    
    def categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Categorize extracted skills by type.
        
        Args:
            skills (List[str]): List of skills to categorize
            
        Returns:
            Dict[str, List[str]]: Categorized skills
        """
        categorized = {
            'technical_skills': [],
            'soft_skills': [],
            'industry_domains': [],
            'certifications': [],
            'methodologies': []
        }
        
        # Technical skills categorization
        tech_skills = self.skills_dict.get('technical_skills', {})
        for category, skill_list in tech_skills.items():
            for skill in skills:
                if skill.lower() in [s.lower() for s in skill_list]:
                    categorized['technical_skills'].append(skill)
        
        # Soft skills categorization
        soft_skills = self.skills_dict.get('soft_skills', {})
        for category, skill_list in soft_skills.items():
            for skill in skills:
                if skill.lower() in [s.lower() for s in skill_list]:
                    categorized['soft_skills'].append(skill)
        
        # Domain categorization
        domains = self.skills_dict.get('industry_domains', [])
        for skill in skills:
            if skill.lower() in [d.lower() for d in domains]:
                categorized['industry_domains'].append(skill)
        
        # Certification categorization
        certs = self.skills_dict.get('certifications', [])
        for skill in skills:
            if skill.lower() in [c.lower() for c in certs]:
                categorized['certifications'].append(skill)
        
        # Methodology categorization
        methods = self.skills_dict.get('methodologies', [])
        for skill in skills:
            if skill.lower() in [m.lower() for m in methods]:
                categorized['methodologies'].append(skill)
        
        # Remove duplicates
        for category in categorized:
            categorized[category] = list(set(categorized[category]))
        
        return categorized
    
    def extract_experience_level(self, text: str) -> str:
        """
        Extract experience level from job text.
        
        Args:
            text (str): Job text to analyze
            
        Returns:
            str: Experience level category
        """
        if not text or not isinstance(text, str):
            return "unknown"
        
        text_lower = text.lower()
        exp_levels = self.skills_dict.get('experience_levels', {})
        
        # Check each experience level category
        for level, keywords in exp_levels.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return level
        
        return "unknown"
    
    def process_job_dataframe(self, df: pd.DataFrame, 
                            text_column: str = 'combined_text') -> pd.DataFrame:
        """
        Process a DataFrame of job data to extract skills.
        
        Args:
            df (pd.DataFrame): DataFrame containing job data
            text_column (str): Column containing text to extract skills from
            
        Returns:
            pd.DataFrame: DataFrame with extracted skills columns
        """
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in DataFrame")
            raise ValueError(f"Column '{text_column}' not found")
        
        df_processed = df.copy()
        
        logger.info(f"Extracting skills from {len(df)} job records")
        
        # Extract skills
        df_processed['extracted_skills'] = df_processed[text_column].apply(
            lambda x: self.extract_skills(x) if pd.notna(x) else []
        )
        
        # Extract experience levels
        df_processed['experience_level'] = df_processed[text_column].apply(
            self.extract_experience_level
        )
        
        # Categorize skills
        df_processed['categorized_skills'] = df_processed['extracted_skills'].apply(
            self.categorize_skills
        )
        
        # Add skill count
        df_processed['skill_count'] = df_processed['extracted_skills'].apply(len)
        
        logger.info("Skill extraction completed successfully")
        return df_processed
    
    def process_job_file(self, input_path: str, output_path: str = None,
                        text_column: str = 'combined_text') -> pd.DataFrame:
        """
        Process a job data file to extract skills.
        
        Args:
            input_path (str): Path to input job data file
            output_path (str): Path to save processed data
            text_column (str): Column containing text to extract skills from
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        try:
            logger.info(f"Loading job data from: {input_path}")
            
            # Load data
            if input_path.endswith('.json'):
                df = pd.read_json(input_path)
            elif input_path.endswith('.csv'):
                df = pd.read_csv(input_path)
            else:
                raise ValueError("Unsupported file format. Use .json or .csv")
            
            # Process skills
            df_processed = self.process_job_dataframe(df, text_column)
            
            # Save if output path specified
            if output_path:
                logger.info(f"Saving processed data to: {output_path}")
                if output_path.endswith('.csv'):
                    df_processed.to_csv(output_path, index=False)
                elif output_path.endswith('.json'):
                    df_processed.to_json(output_path, orient='records', indent=2)
                else:
                    df_processed.to_csv(output_path, index=False)
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error processing job file: {e}")
            raise
    
    def get_skill_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about extracted skills.
        
        Args:
            df (pd.DataFrame): DataFrame with extracted skills
            
        Returns:
            Dict[str, Any]: Skill statistics
        """
        if 'extracted_skills' not in df.columns:
            return {}
        
        all_skills = []
        for skills_list in df['extracted_skills']:
            if isinstance(skills_list, list):
                all_skills.extend(skills_list)
        
        skill_counts = pd.Series(all_skills).value_counts()
        
        stats = {
            'total_jobs': len(df),
            'jobs_with_skills': len(df[df['skill_count'] > 0]),
            'unique_skills': len(skill_counts),
            'avg_skills_per_job': df['skill_count'].mean(),
            'top_10_skills': skill_counts.head(10).to_dict(),
            'experience_level_distribution': df['experience_level'].value_counts().to_dict()
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    try:
        extractor = SkillExtractor()
        
        # Process job data file
        df = extractor.process_job_file(
            input_path="data/processed_jobs/cleaned_jobs.csv",
            output_path="data/processed_jobs/skills_extracted.csv"
        )
        
        print(f"Processed {len(df)} job records")
        
        # Get statistics
        stats = extractor.get_skill_statistics(df)
        print("\nSkill Extraction Statistics:")
        print(f"Jobs with skills: {stats['jobs_with_skills']}/{stats['total_jobs']}")
        print(f"Unique skills found: {stats['unique_skills']}")
        print(f"Average skills per job: {stats['avg_skills_per_job']:.2f}")
        print("\nTop 10 skills:")
        for skill, count in stats['top_10_skills'].items():
            print(f"  {skill}: {count}")
        
    except FileNotFoundError:
        print("Required files not found. Please ensure cleaned job data and skills dictionary exist.")
    except Exception as e:
        print(f"Error: {e}")
