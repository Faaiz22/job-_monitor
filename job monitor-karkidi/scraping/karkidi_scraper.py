"""
Karkidi Job Scraper
Scrapes job listings from karkidi.com
"""

import requests
from bs4 import BeautifulSoup
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import re
from urllib.parse import urljoin, urlparse
import json

from config.settings import SCRAPING_CONFIG


class KarkidiScraper:
    """
    Scraper for Karkidi job listings
    """
    
    def __init__(self):
        self.base_url = SCRAPING_CONFIG['karkidi_base_url']
        self.headers = SCRAPING_CONFIG['headers']
        self.delay = SCRAPING_CONFIG['delay_between_requests']
        self.max_retries = SCRAPING_CONFIG['max_retries']
        self.timeout = SCRAPING_CONFIG['timeout']
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.logger = logging.getLogger('karkidi_scraper')
    
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
    
    def _extract_job_details(self, job_element) -> Optional[Dict]:
        """Extract job details from a job listing element"""
        try:
            job_data = {}
            
            # Extract title
            title_elem = job_element.find('h3', class_='job-title') or job_element.find('h2', class_='job-title') or job_element.find('a', class_='job-link')
            if title_elem:
                job_data['title'] = title_elem.get_text(strip=True)
                
                # Extract job URL if available
                link_elem = title_elem.find('a') if title_elem.name != 'a' else title_elem
                if link_elem and link_elem.get('href'):
                    job_data['url'] = urljoin(self.base_url, link_elem['href'])
            
            # Extract company
            company_elem = job_element.find('div', class_='company-name') or job_element.find('span', class_='company') or job_element.find('p', class_='company')
            if company_elem:
                job_data['company'] = company_elem.get_text(strip=True)
            
            # Extract location
            location_elem = job_element.find('div', class_='location') or job_element.find('span', class_='location') or job_element.find('p', class_='location')
            if location_elem:
                job_data['location'] = location_elem.get_text(strip=True)
            
            # Extract salary if available
            salary_elem = job_element.find('div', class_='salary') or job_element.find('span', class_='salary')
            if salary_elem:
                job_data['salary'] = salary_elem.get_text(strip=True)
            
            # Extract job type
            job_type_elem = job_element.find('div', class_='job-type') or job_element.find('span', class_='job-type')
            if job_type_elem:
                job_data['job_type'] = job_type_elem.get_text(strip=True)
            
            # Extract short description
            desc_elem = job_element.find('div', class_='job-description') or job_element.find('p', class_='description')
            if desc_elem:
                job_data['description'] = desc_elem.get_text(strip=True)[:500]  # Limit to 500 chars
            
            # Extract posting date if available
            date_elem = job_element.find('div', class_='post-date') or job_element.find('span', class_='date')
            if date_elem:
                job_data['posted_date'] = date_elem.get_text(strip=True)
            
            # Add scraping metadata
            job_data['scraped_date'] = datetime.now().isoformat()
            job_data['source'] = 'karkidi.com'
            
            # Only return if we have minimum required data
            if job_data.get('title') and job_data.get('company'):
                return job_data
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting job details: {str(e)}")
            return None
    
    def _get_job_listings_from_page(self, page_url: str) -> List[Dict]:
        """Extract job listings from a single page"""
        self.logger.info(f"Scraping page: {page_url}")
        
        response = self._make_request(page_url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        jobs = []
        
        # Common selectors for job listings
        job_selectors = [
            '.job-item',
            '.job-listing',
            '.job-card',
            '.job-post',
            'article',
            '.listing-item',
            '[class*="job"]'
        ]
        
        job_elements = []
        for selector in job_selectors:
            elements = soup.select(selector)
            if elements:
                job_elements = elements
                break
        
        # If no common selectors work, try to find job listings by content
        if not job_elements:
            # Look for elements containing job-related text
            all_divs = soup.find_all('div')
            job_elements = [
                div for div in all_divs 
                if any(keyword in div.get_text().lower() for keyword in ['job', 'position', 'career', 'vacancy', 'hiring'])
                and div.find('h1', 'h2', 'h3', 'h4')  # Must have a heading
            ][:20]  # Limit to first 20 to avoid noise
        
        self.logger.info(f"Found {len(job_elements)} potential job elements")
        
        for job_element in job_elements:
            job_data = self._extract_job_details(job_element)
            if job_data:
                jobs.append(job_data)
        
        self.logger.info(f"Successfully extracted {len(jobs)} jobs from page")
        
        # Add delay between requests
        time.sleep(self.delay)
        
        return jobs
    
    def _get_next_page_url(self, soup: BeautifulSoup, current_page: int) -> Optional[str]:
        """Get the URL for the next page"""
        # Look for pagination links
        pagination_selectors = [
            f'a[href*="page={current_page + 1}"]',
            f'a[href*="p={current_page + 1}"]',
            'a.next',
            'a[rel="next"]',
            '.pagination a:contains("Next")',
            '.pagination a:contains(">")',
        ]
        
        for selector in pagination_selectors:
            next_link = soup.select_one(selector)
            if next_link and next_link.get('href'):
                return urljoin(self.base_url, next_link['href'])
        
        return None
    
    def scrape_karkidi_jobs(self, max_pages: int = 5, search_terms: List[str] = None) -> List[Dict]:
        """
        Scrape job listings from Karkidi
        
        Args:
            max_pages: Maximum number of pages to scrape
            search_terms: Optional list of search terms to filter jobs
            
        Returns:
            List of job dictionaries
        """
        self.logger.info(f"Starting Karkidi scraping (max_pages: {max_pages})")
        
        all_jobs = []
        
        # Base job search URLs to try
        search_urls = [
            f"{self.base_url}/jobs",
            f"{self.base_url}/job-search",
            f"{self.base_url}/careers",
            f"{self.base_url}/employment",
            f"{self.base_url}/vacancies",
        ]
        
        # Add search term specific URLs if provided
        if search_terms:
            for term in search_terms:
                search_urls.extend([
                    f"{self.base_url}/jobs?q={term}",
                    f"{self.base_url}/search?q={term}",
                    f"{self.base_url}/jobs/search?keyword={term}",
                ])
        
        for base_url in search_urls:
            try:
                # Test if this URL works
                response = self._make_request(base_url)
                if response and response.status_code == 200:
                    self.logger.info(f"Using URL: {base_url}")
                    
                    for page in range(1, max_pages + 1):
                        # Construct page URL
                        if '?' in base_url:
                            page_url = f"{base_url}&page={page}"
                        else:
                            page_url = f"{base_url}?page={page}"
                        
                        jobs = self._get_job_listings_from_page(page_url)
                        
                        if not jobs:
                            self.logger.info(f"No jobs found on page {page}, stopping pagination")
                            break
                        
                        all_jobs.extend(jobs)
                        self.logger.info(f"Total jobs scraped so far: {len(all_jobs)}")
                        
                        # Break if we found jobs (successful scraping)
                        if jobs:
                            break
                    
                    # If we found jobs, no need to try other URLs
                    if all_jobs:
                        break
                        
            except Exception as e:
                self.logger.warning(f"Failed to scrape from {base_url}: {str(e)}")
                continue
        
        # Remove duplicates based on URL or title+company combination
        unique_jobs = []
        seen_jobs = set()
        
        for job in all_jobs:
            # Create unique identifier
            if job.get('url'):
                identifier = job['url']
            else:
                identifier = f"{job.get('title', '')}-{job.get('company', '')}"
            
            if identifier not in seen_jobs:
                seen_jobs.add(identifier)
                unique_jobs.append(job)
        
        self.logger.info(f"Scraping completed. Found {len(unique_jobs)} unique jobs")
        return unique_jobs
    
    def scrape_job_categories(self) -> List[str]:
        """Scrape available job categories from Karkidi"""
        try:
            response = self._make_request(f"{self.base_url}/jobs")
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for category filters or menus
            category_selectors = [
                '.category-filter option',
                '.job-categories a',
                '.category-list li a',
                '[class*="category"] a',
                'select[name*="category"] option'
            ]
            
            categories = []
            for selector in category_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    category = elem.get_text(strip=True)
                    if category and category.lower() not in ['all', 'select category', '']:
                        categories.append(category)
            
            return list(set(categories))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error scraping job categories: {str(e)}")
            return []
    
    def search_jobs_by_keyword(self, keyword: str, max_results: int = 50) -> List[Dict]:
        """Search for jobs by specific keyword"""
        self.logger.info(f"Searching jobs for keyword: {keyword}")
        
        search_urls = [
            f"{self.base_url}/jobs?q={keyword}",
            f"{self.base_url}/search?keyword={keyword}",
            f"{self.base_url}/job-search?q={keyword}",
        ]
        
        all_jobs = []
        
        for url in search_urls:
            try:
                jobs = self._get_job_listings_from_page(url)
                all_jobs.extend(jobs)
                
                if len(all_jobs) >= max_results:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to search from {url}: {str(e)}")
                continue
        
        return all_jobs[:max_results]
    
    def get_job_details(self, job_url: str) -> Optional[Dict]:
        """Get detailed information for a specific job"""
        try:
            response = self._make_request(job_url)
            if not response:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            job_details = {
                'url': job_url,
                'scraped_date': datetime.now().isoformat(),
                'source': 'karkidi.com'
            }
            
            # Extract detailed job information
            title_selectors = ['h1', '.job-title', 'h2.title', '.title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    job_details['title'] = title_elem.get_text(strip=True)
                    break
            
            # Extract company
            company_selectors = ['.company-name', '.company', '[class*="company"]']
            for selector in company_selectors:
                company_elem = soup.select_one(selector)
                if company_elem:
                    job_details['company'] = company_elem.get_text(strip=True)
                    break
            
            # Extract full description
            desc_selectors = ['.job-description', '.description', '.content', '.job-content']
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    job_details['full_description'] = desc_elem.get_text(strip=True)
                    job_details['description'] = job_details['full_description'][:1000]
                    break
            
            # Extract requirements/skills
            req_selectors = ['.requirements', '.skills', '.qualifications']
            for selector in req_selectors:
                req_elem = soup.select_one(selector)
                if req_elem:
                    job_details['requirements'] = req_elem.get_text(strip=True)
                    break
            
            # Extract location
            location_selectors = ['.location', '.job-location', '[class*="location"]']
            for selector in location_selectors:
                location_elem = soup.select_one(selector)
                if location_elem:
                    job_details['location'] = location_elem.get_text(strip=True)
                    break
            
            # Extract salary
            salary_selectors = ['.salary', '.wage', '.compensation', '[class*="salary"]']
            for selector in salary_selectors:
                salary_elem = soup.select_one(selector)
                if salary_elem:
                    job_details['salary'] = salary_elem.get_text(strip=True)
                    break
            
            return job_details
            
        except Exception as e:
            self.logger.error(f"Error getting job details from {job_url}: {str(e)}")
            return None