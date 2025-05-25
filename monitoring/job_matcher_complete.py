"""
Job Matching System
Matches jobs to user profiles based on skills and preferences
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config.settings import MATCHING_CONFIG, SKILLS_PATTERNS


class JobMatcher:
    """
    Matches jobs to user profiles based on skills, categories, and preferences
    """
    
    def __init__(self):
        self.logger = logging.getLogger('job_matcher')
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Load configuration
        self.min_skill_match_score = MATCHING_CONFIG['min_skill_match_score']
        self.min_title_match_score = MATCHING_CONFIG['min_title_match_score']
        self.category_weight = MATCHING_CONFIG['category_weight']
        self.skill_weight = MATCHING_CONFIG['skill_weight']
        self.title_weight = MATCHING_CONFIG['title_weight']
        self.max_jobs_per_notification = MATCHING_CONFIG['max_jobs_per_notification']
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from job description or user profile text"""
        if not text:
            return []
        
        text_lower = text.lower()
        extracted_skills = []
        
        # Use predefined skill patterns
        for skill_category, patterns in SKILLS_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                extracted_skills.extend(matches)
        
        # Additional skill extraction for common technologies
        additional_patterns = [
            r'\b(?:api|rest|graphql|microservices|devops|ci/cd)\b',
            r'\b(?:agile|scrum|kanban|jira|confluence)\b',
            r'\b(?:docker|kubernetes|jenkins|terraform|ansible)\b',
            r'\b(?:pytest|junit|selenium|cypress|jest)\b',
            r'\b(?:git|github|gitlab|bitbucket|svn)\b',
        ]
        
        for pattern in additional_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            extracted_skills.extend(matches)
        
        # Clean and deduplicate
        skills = list(set([skill.strip().lower() for skill in extracted_skills if len(skill.strip()) > 1]))
        
        return skills
    
    def _calculate_skill_match_score(self, user_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill matching score between user and job"""
        if not user_skills or not job_skills:
            return 0.0
        
        user_skills_set = set([skill.lower().strip() for skill in user_skills])
        job_skills_set = set([skill.lower().strip() for skill in job_skills])
        
        # Calculate Jaccard similarity
        intersection = len(user_skills_set.intersection(job_skills_set))
        union = len(user_skills_set.union(job_skills_set))
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        
        # Also calculate overlap percentage from user's perspective
        user_overlap = intersection / len(user_skills_set) if user_skills_set else 0
        
        # Combine both scores
        final_score = (jaccard_score + user_overlap) / 2
        
        return min(final_score, 1.0)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF and cosine similarity"""
        if not text1 or not text2:
            return 0.0
        
        try:
            # Combine texts for fitting vectorizer
            texts = [text1.lower(), text2.lower()]
            
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating text similarity: {str(e)}")
            return 0.0
    
    def _calculate_category_match_score(self, user_categories: List[str], job_category: str) -> float:
        """Calculate category matching score"""
        if not user_categories or not job_category:
            return 0.0
        
        user_categories_lower = [cat.lower().strip() for cat in user_categories]
        job_category_lower = job_category.lower().strip()
        
        # Exact match
        if job_category_lower in user_categories_lower:
            return 1.0
        
        # Partial match - check if any user category is contained in job category or vice versa
        for user_cat in user_categories_lower:
            if user_cat in job_category_lower or job_category_lower in user_cat:
                return 0.7
        
        # Use text similarity for related categories
        user_categories_text = ' '.join(user_categories_lower)
        similarity = self._calculate_text_similarity(user_categories_text, job_category_lower)
        
        return similarity
    
    def _extract_location_preferences(self, user_profile: Dict) -> Dict:
        """Extract location preferences from user profile"""
        preferences = {
            'preferred_locations': user_profile.get('preferred_locations', []),
            'remote_ok': user_profile.get('remote_ok', False),
            'relocation_ok': user_profile.get('relocation_ok', False)
        }
        
        return preferences
    
    def _check_location_match(self, user_profile: Dict, job: Dict) -> bool:
        """Check if job location matches user preferences"""
        location_prefs = self._extract_location_preferences(user_profile)
        job_location = job.get('location', '').lower()
        
        # If user is open to remote work and job allows remote
        if location_prefs['remote_ok'] and job.get('remote_friendly', False):
            return True
        
        # If user is open to relocation
        if location_prefs['relocation_ok']:
            return True
        
        # Check preferred locations
        preferred_locations = [loc.lower() for loc in location_prefs['preferred_locations']]
        
        if not preferred_locations:
            return True  # No location preference specified
        
        # Check if job location matches any preferred location
        for pref_loc in preferred_locations:
            if pref_loc in job_location or job_location in pref_loc:
                return True
        
        return False
    
    def _check_salary_match(self, user_profile: Dict, job: Dict) -> bool:
        """Check if job salary meets user expectations"""
        min_salary = user_profile.get('min_salary')
        job_salary = job.get('salary', '')
        
        if not min_salary or not job_salary:
            return True  # No salary constraints
        
        # Extract salary numbers from job posting
        salary_numbers = re.findall(r'\d{1,3}(?:,\d{3})*', job_salary.replace('$', ''))
        
        if not salary_numbers:
            return True  # Cannot determine salary
        
        # Get the highest salary mentioned
        max_job_salary = max(int(num.replace(',', '')) for num in salary_numbers)
        
        return max_job_salary >= min_salary
    
    def calculate_job_match_score(self, user_profile: Dict, job: Dict) -> float:
        """
        Calculate overall matching score between user and job
        
        Args:
            user_profile: User profile dictionary
            job: Job dictionary
            
        Returns:
            Match score between 0 and 1
        """
        scores = {}
        
        # 1. Skills matching
        user_skills = user_profile.get('skills', [])
        job_text = f"{job.get('description', '')} {job.get('title', '')} {job.get('full_description', '')}"
        job_skills = self._extract_skills_from_text(job_text)
        
        scores['skill_score'] = self._calculate_skill_match_score(user_skills, job_skills)
        
        # 2. Category matching
        user_categories = user_profile.get('interested_categories', [])
        job_category = job.get('predicted_category', '')
        
        scores['category_score'] = self._calculate_category_match_score(user_categories, job_category)
        
        # 3. Title/Role matching
        user_roles = user_profile.get('preferred_roles', [])
        job_title = job.get('title', '')
        
        if user_roles:
            user_roles_text = ' '.join(user_roles)
            scores['title_score'] = self._calculate_text_similarity(user_roles_text, job_title)
        else:
            scores['title_score'] = 0.0
        
        # 4. Calculate weighted final score
        final_score = (
            scores['category_score'] * self.category_weight +
            scores['skill_score'] * self.skill_weight +
            scores['title_score'] * self.title_weight
        )
        
        # Apply filters
        if not self._check_location_match(user_profile, job):
            final_score *= 0.5  # Reduce score for location mismatch
        
        if not self._check_salary_match(user_profile, job):
            final_score *= 0.7  # Reduce score for salary mismatch
        
        return min(final_score, 1.0)
    
    def find_matching_jobs(self, user_profile: Dict, jobs: List[Dict], 
                          min_score: float = None) -> List[Dict]:
        """
        Find jobs that match user profile
        
        Args:
            user_profile: User profile dictionary
            jobs: List of job dictionaries
            min_score: Minimum matching score (uses config default if None)
            
        Returns:
            List of matching jobs with scores
        """
        if min_score is None:
            min_score = max(self.min_skill_match_score, self.min_title_match_score)
        
        matching_jobs = []
        
        for job in jobs:
            try:
                # Calculate match score
                score = self.calculate_job_match_score(user_profile, job)
                
                # Check if job meets minimum score threshold
                if score >= min_score:
                    job_with_score = job.copy()
                    job_with_score['match_score'] = score
                    job_with_score['match_details'] = self._get_match_details(user_profile, job)
                    matching_jobs.append(job_with_score)
                    
            except Exception as e:
                self.logger.error(f"Error calculating match for job {job.get('id', 'unknown')}: {str(e)}")
                continue
        
        # Sort by match score (highest first)
        matching_jobs.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Limit results
        return matching_jobs[:self.max_jobs_per_notification]
    
    def _get_match_details(self, user_profile: Dict, job: Dict) -> Dict:
        """Get detailed matching information for debugging/explanation"""
        details = {}
        
        # Skills matching details
        user_skills = user_profile.get('skills', [])
        job_text = f"{job.get('description', '')} {job.get('title', '')} {job.get('full_description', '')}"
        job_skills = self._extract_skills_from_text(job_text)
        
        user_skills_set = set([skill.lower().strip() for skill in user_skills])
        job_skills_set = set([skill.lower().strip() for skill in job_skills])
        
        details['matched_skills'] = list(user_skills_set.intersection(job_skills_set))
        details['user_skills'] = user_skills
        details['job_skills'] = job_skills
        details['skill_score'] = self._calculate_skill_match_score(user_skills, job_skills)
        
        # Category matching details
        user_categories = user_profile.get('interested_categories', [])
        job_category = job.get('predicted_category', '')
        details['category_score'] = self._calculate_category_match_score(user_categories, job_category)
        details['user_categories'] = user_categories
        details['job_category'] = job_category
        
        # Title matching details
        user_roles = user_profile.get('preferred_roles', [])
        job_title = job.get('title', '')
        if user_roles:
            user_roles_text = ' '.join(user_roles)
            details['title_score'] = self._calculate_text_similarity(user_roles_text, job_title)
        else:
            details['title_score'] = 0.0
        details['user_roles'] = user_roles
        details['job_title'] = job_title
        
        # Location and salary matching
        details['location_match'] = self._check_location_match(user_profile, job)
        details['salary_match'] = self._check_salary_match(user_profile, job)
        
        return details
    
    def get_top_matching_jobs_for_users(self, users: List[Dict], 
                                       jobs: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Get top matching jobs for multiple users
        
        Args:
            users: List of user profile dictionaries (must include 'user_id')
            jobs: List of job dictionaries
            
        Returns:
            Dictionary mapping user_id to list of matching jobs
        """
        user_matches = {}
        
        for user in users:
            user_id = user.get('user_id')
            if not user_id:
                self.logger.warning("User profile missing user_id, skipping")
                continue
            
            try:
                matching_jobs = self.find_matching_jobs(user, jobs)
                user_matches[user_id] = matching_jobs
                
                self.logger.info(f"Found {len(matching_jobs)} matching jobs for user {user_id}")
                
            except Exception as e:
                self.logger.error(f"Error finding matches for user {user_id}: {str(e)}")
                user_matches[user_id] = []
        
        return user_matches
    
    def generate_match_explanation(self, user_profile: Dict, job: Dict) -> str:
        """
        Generate human-readable explanation of why a job matches a user
        
        Args:
            user_profile: User profile dictionary
            job: Job dictionary
            
        Returns:
            Explanation string
        """
        match_details = self._get_match_details(user_profile, job)
        explanations = []
        
        # Skills explanation
        matched_skills = match_details.get('matched_skills', [])
        if matched_skills:
            skills_text = ', '.join(matched_skills[:5])  # Show top 5 skills
            if len(matched_skills) > 5:
                skills_text += f" and {len(matched_skills) - 5} more"
            explanations.append(f"Your skills match: {skills_text}")
        
        # Category explanation
        if match_details.get('category_score', 0) > 0.7:
            explanations.append(f"This {match_details.get('job_category', '')} role aligns with your interests")
        
        # Title explanation
        if match_details.get('title_score', 0) > 0.5:
            explanations.append(f"The job title '{match_details.get('job_title', '')}' matches your preferred roles")
        
        # Location explanation
        if match_details.get('location_match'):
            job_location = job.get('location', '')
            if job.get('remote_friendly'):
                explanations.append("This role offers remote work options")
            elif job_location:
                explanations.append(f"Located in {job_location}")
        
        if explanations:
            return ". ".join(explanations) + "."
        else:
            return "This job matches your profile based on various factors."
    
    def update_user_preferences_from_interactions(self, user_profile: Dict, 
                                                 interactions: List[Dict]) -> Dict:
        """
        Update user preferences based on their interactions with job recommendations
        
        Args:
            user_profile: Current user profile
            interactions: List of interaction dictionaries with keys:
                         'job_id', 'action' (applied/viewed/dismissed), 'timestamp'
            
        Returns:
            Updated user profile
        """
        updated_profile = user_profile.copy()
        
        # Analyze applied jobs to extract preferred skills and categories
        applied_jobs = [i for i in interactions if i.get('action') == 'applied']
        
        if applied_jobs:
            # Extract skills from jobs user applied to
            applied_skills = []
            applied_categories = []
            
            for interaction in applied_jobs:
                job_id = interaction.get('job_id')
                # Note: You would need to fetch the actual job data here
                # This is a placeholder for the logic
                # job = get_job_by_id(job_id)
                # if job:
                #     job_skills = self._extract_skills_from_text(job.get('description', ''))
                #     applied_skills.extend(job_skills)
                #     applied_categories.append(job.get('predicted_category', ''))
            
            # Update user skills with higher weights for applied job skills
            if applied_skills:
                current_skills = set(updated_profile.get('skills', []))
                new_skills = set(applied_skills)
                updated_profile['skills'] = list(current_skills.union(new_skills))
            
            # Update interested categories
            if applied_categories:
                current_categories = set(updated_profile.get('interested_categories', []))
                new_categories = set(applied_categories)
                updated_profile['interested_categories'] = list(current_categories.union(new_categories))
        
        return updated_profile