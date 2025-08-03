"""
Image extraction from URLs and PDFs
"""

import requests
from bs4 import BeautifulSoup
from PIL import Image
import PyPDF2
from pdf2image import convert_from_path
import io
from typing import List, Tuple
from urllib.parse import urljoin, urlparse
import logging

logger = logging.getLogger(__name__)

class ImageExtractor:
    """Extract images from URLs and PDF files"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_from_url(self, url: str) -> List[Tuple[Image.Image, str]]:
        """Extract images from a website URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            img_tags = soup.find_all('img')
            
            images = []
            for i, img_tag in enumerate(img_tags):
                img_url = img_tag.get('src')
                if not img_url:
                    continue
                
                # Handle relative URLs
                img_url = urljoin(url, img_url)
                
                try:
                    img_response = self.session.get(img_url, timeout=10)
                    img_response.raise_for_status()
                    
                    # Open image
                    image = Image.open(io.BytesIO(img_response.content))
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Filter out very small images (likely icons/logos)
                    if image.size[0] >= 100 and image.size[1] >= 100:
                        images.append((image, f"url_{i}_{img_url}"))
                        logger.debug(f"Extracted image {i}: {img_url}")
                
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_url}: {e}")
                    continue
            
            return images
            
        except Exception as e:
            logger.error(f"Failed to extract images from URL {url}: {e}")
            return []
    
    def extract_from_pdf(self, pdf_path: str) -> List[Tuple[Image.Image, str]]:
        """Extract images from a PDF file"""
        try:
            # Convert PDF pages to images
            pages = convert_from_path(pdf_path, dpi=200)
            
            images = []
            for i, page in enumerate(pages):
                # Convert to RGB if necessary
                if page.mode != 'RGB':
                    page = page.convert('RGB')
                
                images.append((page, f"pdf_page_{i}"))
                logger.debug(f"Extracted page {i} from PDF")
            
            return images
            
        except Exception as e:
            logger.error(f"Failed to extract images from PDF {pdf_path}: {e}")
            return []