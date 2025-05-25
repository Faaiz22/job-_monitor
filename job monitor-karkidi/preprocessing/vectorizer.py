"""
Vectorization utilities for job monitoring system.
Uses TF-IDF and skill-based vectorization for job matching and categorization.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillVectorizer:
    """
    Handles vectorization of job descriptions and skills for machine learning.
    """
    
    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.8):
        """
        Initialize the SkillVectorizer.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            min_df (int): Minimum document frequency for TF-IDF
            max_df (float): Maximum document frequency for TF-IDF
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),  # Include unigrams and bigrams
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.skill_binarizer = MultiLabelBinarizer()
        
        # Fitted flags
        self.tfidf_fitted = False
        self.skill_fitted = False
        
        # Feature names
        self.tfidf_feature_names = []
        self.skill_feature_names = []
    
    def fit_transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform texts.
        
        Args:
            texts (List[str]): List of texts to vectorize
            
        Returns:
            np.ndarray: TF-IDF matrix
        """
        try:
            logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} documents")
            
            # Filter out empty or invalid texts
            valid_texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]
            
            # Fit and transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_texts)
            self.tfidf_fitted = True
            self.tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
            
            logger.info(f"TF-IDF vectorizer fitted with {len(self.tfidf_feature_names)} features")
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            
            return tfidf_matrix.toarray()
            
        except Exception as e:
            logger.error(f"Error transforming skill lists: {e}")
            raise
    
    def create_combined_vectors(self, tfidf_matrix: np.ndarray, 
                              skill_matrix: np.ndarray,
                              tfidf_weight: float = 0.7,
                              skill_weight: float = 0.3) -> np.ndarray:
        """
        Combine TF-IDF and skill vectors with specified weights.
        
        Args:
            tfidf_matrix (np.ndarray): TF-IDF feature matrix
            skill_matrix (np.ndarray): Skill feature matrix
            tfidf_weight (float): Weight for TF-IDF features
            skill_weight (float): Weight for skill features
            
        Returns:
            np.ndarray: Combined feature matrix
        """
        try:
            # Normalize weights
            total_weight = tfidf_weight + skill_weight
            tfidf_weight = tfidf_weight / total_weight
            skill_weight = skill_weight / total_weight
            
            # Weight the matrices
            weighted_tfidf = tfidf_matrix * tfidf_weight
            weighted_skills = skill_matrix * skill_weight
            
            # Combine horizontally
            combined_matrix = np.hstack([weighted_tfidf, weighted_skills])
            
            logger.info(f"Combined matrix shape: {combined_matrix.shape}")
            logger.info(f"TF-IDF weight: {tfidf_weight:.2f}, Skill weight: {skill_weight:.2f}")
            
            return combined_matrix
            
        except Exception as e:
            logger.error(f"Error combining vectors: {e}")
            raise
    
    def fit_transform(self, df: pd.DataFrame, 
                     text_column: str = 'combined_text',
                     skill_column: str = 'extracted_skills') -> Dict[str, np.ndarray]:
        """
        Fit all vectorizers and transform data.
        
        Args:
            df (pd.DataFrame): DataFrame containing job data
            text_column (str): Column containing text data
            skill_column (str): Column containing skill lists
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing different vector representations
        """
        try:
            logger.info(f"Fitting vectorizers on {len(df)} records")
            
            # Extract text and skills
            texts = df[text_column].fillna('').tolist()
            skills = df[skill_column].fillna('').apply(
                lambda x: x if isinstance(x, list) else []
            ).tolist()
            
            # Fit and transform TF-IDF
            tfidf_matrix = self.fit_transform_tfidf(texts)
            
            # Fit and transform skills
            skill_matrix = self.fit_transform_skills(skills)
            
            # Create combined vectors
            combined_matrix = self.create_combined_vectors(tfidf_matrix, skill_matrix)
            
            vectors = {
                'tfidf': tfidf_matrix,
                'skills': skill_matrix,
                'combined': combined_matrix
            }
            
            logger.info("Vectorization completed successfully")
            return vectors
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {e}")
            raise
    
    def transform(self, df: pd.DataFrame,
                 text_column: str = 'combined_text',
                 skill_column: str = 'extracted_skills') -> Dict[str, np.ndarray]:
        """
        Transform data using fitted vectorizers.
        
        Args:
            df (pd.DataFrame): DataFrame containing job data
            text_column (str): Column containing text data
            skill_column (str): Column containing skill lists
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing different vector representations
        """
        try:
            logger.info(f"Transforming {len(df)} records")
            
            # Extract text and skills
            texts = df[text_column].fillna('').tolist()
            skills = df[skill_column].fillna('').apply(
                lambda x: x if isinstance(x, list) else []
            ).tolist()
            
            # Transform TF-IDF
            tfidf_matrix = self.transform_tfidf(texts)
            
            # Transform skills
            skill_matrix = self.transform_skills(skills)
            
            # Create combined vectors
            combined_matrix = self.create_combined_vectors(tfidf_matrix, skill_matrix)
            
            vectors = {
                'tfidf': tfidf_matrix,
                'skills': skill_matrix,
                'combined': combined_matrix
            }
            
            return vectors
            
        except Exception as e:
            logger.error(f"Error in transform: {e}")
            raise
    
    def calculate_similarity(self, vectors1: np.ndarray, 
                           vectors2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between two sets of vectors.
        
        Args:
            vectors1 (np.ndarray): First set of vectors
            vectors2 (np.ndarray): Second set of vectors
            
        Returns:
            np.ndarray: Similarity matrix
        """
        try:
            similarity_matrix = cosine_similarity(vectors1, vectors2)
            return similarity_matrix
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise
    
    def find_similar_jobs(self, query_vector: np.ndarray,
                         job_vectors: np.ndarray,
                         top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find most similar jobs to a query vector.
        
        Args:
            query_vector (np.ndarray): Query vector (1D)
            job_vectors (np.ndarray): Job vectors matrix
            top_k (int): Number of top matches to return
            
        Returns:
            List[Tuple[int, float]]: List of (index, similarity_score) tuples
        """
        try:
            # Reshape query vector if needed
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, job_vectors)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return indices with scores
            results = [(int(idx), float(similarities[idx])) for idx in top_indices]
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar jobs: {e}")
            raise
    
    def save_vectorizer(self, filepath: str):
        """
        Save fitted vectorizers to file.
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            vectorizer_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer if self.tfidf_fitted else None,
                'skill_binarizer': self.skill_binarizer if self.skill_fitted else None,
                'tfidf_fitted': self.tfidf_fitted,
                'skill_fitted': self.skill_fitted,
                'tfidf_feature_names': self.tfidf_feature_names,
                'skill_feature_names': self.skill_feature_names,
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df
            }
            
            joblib.dump(vectorizer_data, filepath)
            logger.info(f"Vectorizer saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving vectorizer: {e}")
            raise
    
    def load_vectorizer(self, filepath: str):
        """
        Load fitted vectorizers from file.
        
        Args:
            filepath (str): Path to load the vectorizer from
        """
        try:
            vectorizer_data = joblib.load(filepath)
            
            self.tfidf_vectorizer = vectorizer_data.get('tfidf_vectorizer')
            self.skill_binarizer = vectorizer_data.get('skill_binarizer')
            self.tfidf_fitted = vectorizer_data.get('tfidf_fitted', False)
            self.skill_fitted = vectorizer_data.get('skill_fitted', False)
            self.tfidf_feature_names = vectorizer_data.get('tfidf_feature_names', [])
            self.skill_feature_names = vectorizer_data.get('skill_feature_names', [])
            self.max_features = vectorizer_data.get('max_features', 5000)
            self.min_df = vectorizer_data.get('min_df', 2)
            self.max_df = vectorizer_data.get('max_df', 0.8)
            
            logger.info(f"Vectorizer loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading vectorizer: {e}")
            raise
    
    def get_feature_importance(self, vector: np.ndarray,
                             vector_type: str = 'combined') -> Dict[str, float]:
        """
        Get feature importance for a given vector.
        
        Args:
            vector (np.ndarray): Input vector
            vector_type (str): Type of vector ('tfidf', 'skills', 'combined')
            
        Returns:
            Dict[str, float]: Feature names and their importance scores
        """
        try:
            feature_importance = {}
            
            if vector_type == 'tfidf' and self.tfidf_fitted:
                for i, importance in enumerate(vector[:len(self.tfidf_feature_names)]):
                    if importance > 0:
                        feature_importance[self.tfidf_feature_names[i]] = float(importance)
            
            elif vector_type == 'skills' and self.skill_fitted:
                for i, importance in enumerate(vector[:len(self.skill_feature_names)]):
                    if importance > 0:
                        feature_importance[self.skill_feature_names[i]] = float(importance)
            
            elif vector_type == 'combined':
                # TF-IDF features
                tfidf_end = len(self.tfidf_feature_names)
                for i, importance in enumerate(vector[:tfidf_end]):
                    if importance > 0:
                        feature_importance[f"tfidf_{self.tfidf_feature_names[i]}"] = float(importance)
                
                # Skill features
                for i, importance in enumerate(vector[tfidf_end:tfidf_end + len(self.skill_feature_names)]):
                    if importance > 0:
                        feature_importance[f"skill_{self.skill_feature_names[i]}"] = float(importance)
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def process_job_file(self, input_path: str, 
                        output_path: str = None,
                        save_vectors: bool = True) -> pd.DataFrame:
        """
        Process a job data file with vectorization.
        
        Args:
            input_path (str): Path to input job data file
            output_path (str): Path to save processed data
            save_vectors (bool): Whether to save vector representations
            
        Returns:
            pd.DataFrame: Processed DataFrame with vectors
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
            
            # Fit and transform vectors
            vectors = self.fit_transform(df)
            
            # Add vector information to DataFrame (optional, for small datasets)
            if len(df) < 1000:  # Only for smaller datasets to avoid memory issues
                df['tfidf_vector'] = [vec.tolist() for vec in vectors['tfidf']]
                df['skill_vector'] = [vec.tolist() for vec in vectors['skills']]
            
            # Save vectors separately if requested
            if save_vectors and output_path:
                vector_path = output_path.replace('.csv', '_vectors.npz').replace('.json', '_vectors.npz')
                np.savez_compressed(vector_path, **vectors)
                logger.info(f"Vectors saved to {vector_path}")
            
            # Save vectorizer
            if output_path:
                vectorizer_path = output_path.replace('.csv', '_vectorizer.pkl').replace('.json', '_vectorizer.pkl')
                self.save_vectorizer(vectorizer_path)
            
            # Save processed DataFrame
            if output_path:
                logger.info(f"Saving processed data to: {output_path}")
                if output_path.endswith('.csv'):
                    df.to_csv(output_path, index=False)
                elif output_path.endswith('.json'):
                    df.to_json(output_path, orient='records', indent=2)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing job file: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    try:
        vectorizer = SkillVectorizer(max_features=3000, min_df=2, max_df=0.8)
        
        # Process job data file
        df = vectorizer.process_job_file(
            input_path="data/processed_jobs/skills_extracted.csv",
            output_path="data/processed_jobs/vectorized_jobs.csv"
        )
        
        print(f"Processed {len(df)} job records")
        print(f"TF-IDF features: {len(vectorizer.tfidf_feature_names)}")
        print(f"Skill features: {len(vectorizer.skill_feature_names)}")
        
        # Save vectorizer to models directory
        vectorizer.save_vectorizer("models/tfidf_vectorizer.pkl")
        print("Vectorizer saved to models/tfidf_vectorizer.pkl")
        
    except FileNotFoundError:
        print("Required files not found. Please ensure skill-extracted job data exists.")
    except Exception as e:
        print(f"Error: {e}")error(f"Error fitting TF-IDF vectorizer: {e}")
            raise
    
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted TF-IDF vectorizer.
        
        Args:
            texts (List[str]): List of texts to transform
            
        Returns:
            np.ndarray: TF-IDF matrix
        """
        if not self.tfidf_fitted:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_transform_tfidf first.")
        
        try:
            # Filter out empty or invalid texts
            valid_texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]
            
            tfidf_matrix = self.tfidf_vectorizer.transform(valid_texts)
            return tfidf_matrix.toarray()
            
        except Exception as e:
            logger.error(f"Error transforming texts with TF-IDF: {e}")
            raise
    
    def fit_transform_skills(self, skill_lists: List[List[str]]) -> np.ndarray:
        """
        Fit skill binarizer and transform skill lists.
        
        Args:
            skill_lists (List[List[str]]): List of skill lists
            
        Returns:
            np.ndarray: Binary skill matrix
        """
        try:
            logger.info(f"Fitting skill binarizer on {len(skill_lists)} skill lists")
            
            # Filter out invalid skill lists
            valid_skill_lists = []
            for skills in skill_lists:
                if isinstance(skills, list):
                    valid_skill_lists.append([skill.lower() for skill in skills if isinstance(skill, str)])
                else:
                    valid_skill_lists.append([])
            
            # Fit and transform
            skill_matrix = self.skill_binarizer.fit_transform(valid_skill_lists)
            self.skill_fitted = True
            self.skill_feature_names = self.skill_binarizer.classes_.tolist()
            
            logger.info(f"Skill binarizer fitted with {len(self.skill_feature_names)} unique skills")
            logger.info(f"Skill matrix shape: {skill_matrix.shape}")
            
            return skill_matrix
            
        except Exception as e:
            logger.error(f"Error fitting skill binarizer: {e}")
            raise
    
    def transform_skills(self, skill_lists: List[List[str]]) -> np.ndarray:
        """
        Transform skill lists using fitted binarizer.
        
        Args:
            skill_lists (List[List[str]]): List of skill lists
            
        Returns:
            np.ndarray: Binary skill matrix
        """
        if not self.skill_fitted:
            raise ValueError("Skill binarizer not fitted. Call fit_transform_skills first.")
        
        try:
            # Filter out invalid skill lists
            valid_skill_lists = []
            for skills in skill_lists:
                if isinstance(skills, list):
                    valid_skill_lists.append([skill.lower() for skill in skills if isinstance(skill, str)])
                else:
                    valid_skill_lists.append([])
            
            skill_matrix = self.skill_binarizer.transform(valid_skill_lists)
            return skill_matrix
            
        except Exception as e:
            logger.