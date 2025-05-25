"""
External Job Scraper
Scrapes job descriptions from external URLs
"""

import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import Optional, Dict
import re
from urllib.parse import urlparse

from config.settings import SCRAPING_CONFIG


class ExternalScraper:
    """
    Scraper for external job description URLs
    Enhances job data with full descriptions from job posting URLs
    """
    
    def __init__(self):
        self.headers = SCRAPING_CONFIG['headers']
        self.delay = SCRAPING_CONFIG['delay_between_requests']
        self.max_retries = SCRAPING_CONFIG['max_retries']
        self.timeout = SCRAPING_CONFIG['timeout']
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.logger = logging.getLogger('external_scraper')
    
    def _make_request(self, url: str, retries: int = 0) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            if retries < self.max_retries:
                self.logger.warning(f"Request failed, retrying... ({retries + 1}/{self.max_retries}): {str(e)}")
                time.sleep(self.delay * (retries + 1))
                return self._make_request(url, retries + 1)
            else:
                self.logger.error(f"Request failed after {self.max_retries} retries: {str(e)}")
                return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common web artifacts
        text = re.sub(r'Share\s*this\s*job', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Apply\s*now', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Save\s*job', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Print\s*job', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_job_content(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract job description content from HTML"""
        try:
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Common selectors for job descriptions
            content_selectors = [
                '.job-description',
                '.job-content',
                '.description',
                '.content',
                '.job-details',
                '.job-info',
                '#job-description',
                '#description',
                '[class*="description"]',
                '[class*="content"]',
                '[class*="detail"]',
                'article',
                'main',
                '.main-content'
            ]
            
            content = None
            
            # Try each selector
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Get the largest element (likely the main content)
                    content_elem = max(elements, key=lambda x: len(x.get_text()))
                    content = content_elem.get_text(separator=' ', strip=True)
                    if len(content) > 100:  # Minimum content length
                        break
            
            # If no specific selectors work, try to extract from body
            if not content or len(content) < 100:
                body = soup.find('body')
                if body:
                    # Remove navigation and sidebar elements
                    for unwanted in body.select('nav, .nav, .navigation, .sidebar, .menu, .header, .footer'):
                        unwanted.decompose()
                    
                    content = body.get_text(separator=' ', strip=True)
            
            # Clean and validate content
            if content:
                content = self._clean_text(content)
                
                # Basic validation - should contain job-related keywords
                job_keywords = ['job', 'position', 'role', 'responsibilities', 'requirements', 
                               'experience', 'skills', 'qualifications', 'candidate', 'apply']
                
                content_lower = content.lower()
                keyword_count = sum(1 for keyword in job_keywords if keyword in content_lower)
                
                if keyword_count >= 3 and len(content) >= 200:
                    return content[:5000]  # Limit to 5000 characters
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting job content from {url}: {str(e)}")
            return None
    
    def scrape_job_description(self, url: str) -> Optional[str]:
        """
        Scrape full job description from a URL
        
        Args:
            url: Job posting URL
            
        Returns:
            Full job description text or None if failed
        """
        if not url or not url.startswith(('http://', 'https://')):
            return None
        
        try:
            self.logger.debug(f"Scraping job description from: {url}")
            
            response = self._make_request(url)
            if not response:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            content = self._extract_job_content(soup, url)
            
            if content:
                self.logger.debug(f"Successfully extracted {len(content)} characters from {url}")
            else:
                self.logger.warning(f"No content extracted from {url}")
            
            # Add delay between requests
            time.sleep(self.delay)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error scraping job description from {url}: {str(e)}")
            return None
    
    def enhance_job_data(self, job_data: Dict) -> Dict:
        """
        Enhance job data with full description from URL
        
        Args:
            job_data: Job dictionary with URL
            
        Returns:
            Enhanced job dictionary
        """
        if not job_data.get('url'):
            return job_data
        
        try:
            full_description = self.scrape_job_description(job_data['url'])
            
            if full_description:
                # Add full description
                job_data['full_description'] = full_description
                
                # Update short description if not present or too short
                if not job_data.get('description') or len(job_data['description']) < 100:
                    job_data['description'] = full_description[:1000]
                
                # Extract additional information from full description
                self._extract_additional_info(job_data, full_description)
            
        except Exception as e:
            self.logger.error(f"Error enhancing job data: {str(e)}")
        
        return job_data
    
    def _extract_additional_info(self, job_data: Dict, full_description: str):
        """Extract additional information from full job description"""
        try:
            desc_lower = full_description.lower()
            
            # Extract salary information if not present
            if not job_data.get('salary'):
                salary_patterns = [
                    r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:-|to)\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per|/)\s*(?:hour|hr|year|yr|month)',
                    r'(\d{1,3}(?:,\d{3})*)\s*(?:-|to)\s*(\d{1,3}(?:,\d{3})*)\s*(?:per|/)\s*(?:hour|year|month)',
                ]
                
                for pattern in salary_patterns:
                    match = re.search(pattern, full_description, re.IGNORECASE)
                    if match:
                        job_data['salary'] = match.group(0)
                        break
            
            # Extract job type if not present
            if not job_data.get('job_type'):
                job_type_patterns = [
                    r'\b(full[- ]?time|part[- ]?time|contract|temporary|permanent|internship|freelance)\b'
                ]
                
                for pattern in job_type_patterns:
                    match = re.search(pattern, desc_lower)
                    if match:
                        job_data['job_type'] = match.group(1).title()
                        break
            
            # Extract experience level
            experience_patterns = [
                r'(\d+)(?:\+|\s*or\s*more)?\s*years?\s*(?:of\s*)?experience',
                r'entry[- ]level',
                r'senior[- ]level',
                r'mid[- ]level',
                r'junior[- ]level'
            ]
            
            for pattern in experience_patterns:
                match = re.search(pattern, desc_lower)
                if match:
                    job_data['experience_level'] = match.group(0)
                    break
            
            # Extract remote work information
            remote_keywords = ['remote', 'work from home', 'telecommute', 'distributed', 'virtual']
            if any(keyword in desc_lower for keyword in remote_keywords):
                job_data['remote_friendly'] = True
            
        except Exception as e:
            self.logger.error(f"Error extracting additional info: {str(e)}")
    
    def batch_enhance_jobs(self, jobs: list, max_jobs: int = None) -> list:
        """
        Enhance multiple jobs with full descriptions
        
        Args:
            jobs: List of job dictionaries
            max_jobs: Maximum number of jobs to enhance (None for all)
            
        Returns:
            List of enhanced job dictionaries
        """
        if max_jobs:
            jobs = jobs[:max_jobs]
        
        enhanced_jobs = []
        
        for i, job in enumerate(jobs):
            try:
                self.logger.info(f"Enhancing job {i+1}/{len(jobs)}: {job.get('title', 'Unknown')}")
                enhanced_job = self.enhance_job_data(job.copy())
                enhanced_jobs.append(enhanced_job)
                
                # Add delay between requests
                if i < len(jobs) - 1:
                    time.sleep(self.delay)
                    
            except Exception as e:
                self.logger.error(f"Error enhancing job {i+1}: {str(e)}")
                enhanced_jobs.append(job)  # Add original job if enhancement fails
        
        return enhanced_jobs
    
    def validate_url(self, url: str) -> bool:
        """Validate if URL is accessible and contains job content"""
        try:
            response = self._make_request(url)
            if not response or response.status_code != 200:
                return False
            
            soup = BeautifulSoup(response.content, 'html.parser')
            content = self._extract_job_content(soup, url)
            
            return content is not None and len(content) > 100
            
        except Exception:
            return False
