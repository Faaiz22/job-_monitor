"""
Preprocessing module for job monitoring system.
Contains text cleaning, skill extraction, and vectorization utilities.
"""

from .text_cleaner import TextCleaner
from .skill_extractor import SkillExtractor
from .vectorizer import SkillVectorizer

__version__ = "1.0.0"
__author__ = "Job Monitoring System"

__all__ = [
    'TextCleaner',
    'SkillExtractor', 
    'SkillVectorizer'
]

# Module-level convenience functions
def preprocess_job_data(input_path: str, 
                       output_path: str = None,
                       skills_dict_path: str = "data/skills_dictionary.json",
                       save_intermediate: bool = True):
    """
    Complete preprocessing pipeline for job data.
    
    Args:
        input_path (str): Path to raw job data file
        output_path (str): Path to save final processed data
        skills_dict_path (str): Path to skills dictionary
        save_intermediate (bool): Whether to save intermediate processing steps
        
    Returns:
        tuple: (processed_dataframe, vectors_dict, vectorizer_instance)
    """
    import pandas as pd
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting complete job data preprocessing pipeline")
        
        # Determine output directory
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = Path(output_path).stem
        else:
            output_dir = Path("data/processed_jobs")
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = "processed_jobs"
        
        # Step 1: Text Cleaning
        logger.info("Step 1: Cleaning text data")
        cleaner = TextCleaner()
        
        if save_intermediate:
            cleaned_path = output_dir / f"{base_name}_cleaned.csv"
        else:
            cleaned_path = None
            
        df_cleaned = cleaner.process_job_file(input_path, cleaned_path)
        logger.info(f"Text cleaning completed. Shape: {df_cleaned.shape}")
        
        # Step 2: Skill Extraction
        logger.info("Step 2: Extracting skills")
        extractor = SkillExtractor(skills_dict_path)
        
        if save_intermediate:
            skills_path = output_dir / f"{base_name}_skills.csv"
        else:
            skills_path = None
            
        df_skills = extractor.process_job_dataframe(df_cleaned)
        
        if skills_path:
            df_skills.to_csv(skills_path, index=False)
            logger.info(f"Skills extracted and saved to {skills_path}")
        
        # Get skill statistics
        stats = extractor.get_skill_statistics(df_skills)
        logger.info(f"Skill extraction completed. Found {stats['unique_skills']} unique skills")
        
        # Step 3: Vectorization
        logger.info("Step 3: Creating vector representations")
        vectorizer = SkillVectorizer(max_features=3000, min_df=2, max_df=0.8)
        
        vectors = vectorizer.fit_transform(df_skills)
        logger.info(f"Vectorization completed. TF-IDF: {vectors['tfidf'].shape}, "
                   f"Skills: {vectors['skills'].shape}, Combined: {vectors['combined'].shape}")
        
        # Save final processed data
        if output_path:
            df_skills.to_csv(output_path, index=False)
            logger.info(f"Final processed data saved to {output_path}")
            
            # Save vectors
            vector_path = str(output_path).replace('.csv', '_vectors.npz')
            import numpy as np
            np.savez_compressed(vector_path, **vectors)
            logger.info(f"Vectors saved to {vector_path}")
            
            # Save vectorizer
            vectorizer_path = str(output_path).replace('.csv', '_vectorizer.pkl')
            vectorizer.save_vectorizer(vectorizer_path)
            logger.info(f"Vectorizer saved to {vectorizer_path}")
        
        logger.info("Complete preprocessing pipeline finished successfully")
        
        return df_skills, vectors, vectorizer
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise


def load_processed_data(data_path: str, 
                       vectors_path: str = None,
                       vectorizer_path: str = None):
    """
    Load previously processed job data.
    
    Args:
        data_path (str): Path to processed job data CSV
        vectors_path (str): Path to vectors NPZ file
        vectorizer_path (str): Path to vectorizer PKL file
        
    Returns:
        tuple: (dataframe, vectors_dict, vectorizer_instance)
    """
    import pandas as pd
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load DataFrame
        df = pd.read_csv(data_path)
        logger.info(f"Loaded processed data: {df.shape}")
        
        vectors = None
        vectorizer = None
        
        # Load vectors if path provided
        if vectors_path:
            vectors_data = np.load(vectors_path)
            vectors = {key: vectors_data[key] for key in vectors_data.files}
            logger.info(f"Loaded vectors from {vectors_path}")
        
        # Load vectorizer if path provided
        if vectorizer_path:
            vectorizer = SkillVectorizer()
            vectorizer.load_vectorizer(vectorizer_path)
            logger.info(f"Loaded vectorizer from {vectorizer_path}")
        
        return df, vectors, vectorizer
        
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise


def get_processing_statistics(df: pd.DataFrame) -> dict:
    """
    Get comprehensive statistics about processed job data.
    
    Args:
        df (pd.DataFrame): Processed job DataFrame
        
    Returns:
        dict: Processing statistics
    """
    stats = {}
    
    try:
        # Basic statistics
        stats['total_jobs'] = len(df)
        stats['columns'] = list(df.columns)
        
        # Text cleaning statistics
        if 'combined_text' in df.columns:
            stats['avg_text_length'] = df['combined_text'].str.len().mean()
            stats['empty_text_count'] = df['combined_text'].isna().sum()
        
        # Skill extraction statistics
        if 'extracted_skills' in df.columns:
            stats['jobs_with_skills'] = (df['skill_count'] > 0).sum()
            stats['avg_skills_per_job'] = df['skill_count'].mean()
            stats['max_skills_per_job'] = df['skill_count'].max()
            
            # Get unique skills
            all_skills = []
            for skills_list in df['extracted_skills']:
                if isinstance(skills_list, list):
                    all_skills.extend(skills_list)
            
            skill_counts = pd.Series(all_skills).value_counts()
            stats['unique_skills'] = len(skill_counts)
            stats['top_10_skills'] = skill_counts.head(10).to_dict()
        
        # Experience level distribution
        if 'experience_level' in df.columns:
            stats['experience_distribution'] = df['experience_level'].value_counts().to_dict()
        
        # Missing data analysis
        stats['missing_data'] = df.isnull().sum().to_dict()
        
        return stats
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error getting processing statistics: {e}")
        return stats


# Configuration constants
DEFAULT_SKILLS_DICT_PATH = "data/skills_dictionary.json"
DEFAULT_OUTPUT_DIR = "data/processed_jobs"
DEFAULT_MODELS_DIR = "models"

# Processing configuration
PROCESSING_CONFIG = {
    'text_cleaning': {
        'remove_html': True,
        'remove_urls': True,
        'normalize_text': True,
        'combine_text_columns': True
    },
    'skill_extraction': {
        'use_spacy': True,
        'extract_experience_level': True,
        'categorize_skills': True
    },
    'vectorization': {
        'max_features': 3000,
        'min_df': 2,
        'max_df': 0.8,
        'ngram_range': (1, 2),
        'tfidf_weight': 0.7,
        'skill_weight': 0.3
    }
}