"""
Utility functions for the Data Analyst Agent
"""

import re
import os
import tempfile
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO, StringIO
import json
import requests
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileHandler:
    """Handles file operations and conversions"""
    
    @staticmethod
    def save_uploaded_file(file, filename: str) -> str:
        """Save uploaded file to temporary directory and return path"""
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        return file_path
    
    @staticmethod
    def read_file_content(file) -> str:
        """Read content from uploaded file"""
        try:
            return file.read().decode('utf-8')
        except UnicodeDecodeError:
            # Try different encodings
            file.seek(0)
            try:
                return file.read().decode('latin-1')
            except:
                file.seek(0)
                return file.read().decode('utf-8', errors='ignore')
    
    @staticmethod
    def detect_file_type(filename: str) -> str:
        """Detect file type based on extension"""
        ext = filename.lower().split('.')[-1]
        file_types = {
            'csv': 'csv',
            'xlsx': 'excel',
            'xls': 'excel',
            'json': 'json',
            'txt': 'text',
            'pdf': 'pdf',
            'png': 'image',
            'jpg': 'image',
            'jpeg': 'image'
        }
        return file_types.get(ext, 'unknown')

class DataCleaner:
    """Data cleaning and preprocessing utilities"""
    
    @staticmethod
    def clean_numeric_column(series: pd.Series, remove_currency=True, remove_commas=True) -> pd.Series:
        """Clean numeric data by removing currency symbols, commas, etc."""
        if series.dtype == 'object':
            # Convert to string first
            cleaned = series.astype(str)
            
            if remove_currency:
                # Remove currency symbols
                cleaned = cleaned.str.replace(r'[\$£€¥₹]', '', regex=True)
            
            if remove_commas:
                # Remove commas and spaces
                cleaned = cleaned.str.replace(',', '').str.replace(' ', '')
            
            # Extract first number found (handles cases like "$1.2 billion")
            numeric_pattern = r'(\d+\.?\d*)'
            extracted = cleaned.str.extract(numeric_pattern)[0]
            
            # Handle billions, millions, thousands
            multipliers = {
                'billion': 1000,
                'million': 1,
                'thousand': 0.001,
                'b': 1000,
                'm': 1,
                'k': 0.001
            }
            
            result = pd.to_numeric(extracted, errors='coerce')
            
            # Apply multipliers if text contains them
            for key, multiplier in multipliers.items():
                mask = cleaned.str.contains(key, case=False, na=False)
                result.loc[mask] *= multiplier
            
            return result
        
        return pd.to_numeric(series, errors='coerce')
    
    @staticmethod
    def extract_year_from_text(series: pd.Series) -> pd.Series:
        """Extract 4-digit years from text"""
        year_pattern = r'(\d{4})'
        return pd.to_numeric(series.astype(str).str.extract(year_pattern)[0], errors='coerce')
    
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names by removing special characters and standardizing"""
        df = df.copy()
        df.columns = df.columns.astype(str)
        df.columns = df.columns.str.lower().str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
        df.columns = df.columns.str.replace(r'_+', '_', regex=True).str.strip('_')
        return df

class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def extract_questions(text: str) -> List[str]:
        """Extract numbered questions from text"""
        lines = text.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            # Match patterns like "1.", "1)", "Question 1:", etc.
            if re.match(r'^\d+[\.\):]', line) or re.match(r'^Question\s+\d+', line, re.IGNORECASE):
                questions.append(line)
        
        return questions
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'https?://[^\s<>"{\'}|\\^`\[\]]+'
        return re.findall(url_pattern, text)
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict]:
        """Extract JSON object from text"""
        try:
            # Look for JSON patterns
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
            
            return None
        except:
            return None

class VisualizationUtils:
    """Utilities for creating visualizations"""
    
    @staticmethod
    def create_base64_plot(fig, format='png', dpi=100, max_size_kb=100) -> str:
        """Convert matplotlib figure to base64 string with size limit"""
        buffer = BytesIO()
        
        # Try different quality settings to stay under size limit
        for current_dpi in [dpi, 80, 60, 40]:
            buffer.seek(0)
            buffer.truncate()
            
            try:
                fig.savefig(buffer, format=format, dpi=current_dpi, 
                           bbox_inches='tight', facecolor='white')
                buffer.seek(0)
                
                # Check size
                size_kb = len(buffer.getvalue()) / 1024
                
                if size_kb <= max_size_kb:
                    plot_data = base64.b64encode(buffer.getvalue()).decode()
                    plt.close(fig)
                    return f"data:image/{format};base64,{plot_data}"
                    
            except Exception as e:
                logger.error(f"Error creating plot at DPI {current_dpi}: {e}")
        
        # If still too large, return empty string
        plt.close(fig)
        return ""
    
    @staticmethod
    def setup_plot_style():
        """Set up consistent plot styling"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })

class StatisticsUtils:
    """Statistical analysis utilities"""
    
    @staticmethod
    def safe_correlation(x: pd.Series, y: pd.Series) -> float:
        """Calculate correlation safely, handling NaN values"""
        try:
            # Remove NaN values
            mask = ~(pd.isna(x) | pd.isna(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 2:
                return 0.0
                
            return x_clean.corr(y_clean)
        except:
            return 0.0
    
    @staticmethod
    def safe_regression(x: pd.Series, y: pd.Series) -> Tuple[float, float, float]:
        """Perform linear regression safely, return (slope, intercept, r_squared)"""
        from scipy import stats
        
        try:
            # Remove NaN values
            mask = ~(pd.isna(x) | pd.isna(y))
            x_clean = x[mask].astype(float)
            y_clean = y[mask].astype(float)
            
            if len(x_clean) < 2:
                return 0.0, 0.0, 0.0
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            return slope, intercept, r_value**2
            
        except:
            return 0.0, 0.0, 0.0

class WebScrapingUtils:
    """Web scraping utilities"""
    
    @staticmethod
    def safe_request(url: str, timeout=30) -> Optional[requests.Response]:
        """Make HTTP request safely with proper headers"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None
    
    @staticmethod
    def extract_tables_from_html(html_content: str) -> List[pd.DataFrame]:
        """Extract all tables from HTML content"""
        try:
            tables = pd.read_html(StringIO(html_content))
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return []

class DateUtils:
    """Date and time utilities"""
    
    @staticmethod
    def parse_date_flexible(date_str: str) -> Optional[datetime]:
        """Parse dates in various formats"""
        if pd.isna(date_str) or not isinstance(date_str, str):
            return None
            
        date_formats = [
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%B %d, %Y',
            '%d %B %Y',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except:
                continue
        
        return None
    
    @staticmethod
    def calculate_days_difference(date1_str: str, date2_str: str) -> Optional[int]:
        """Calculate difference in days between two date strings"""
        date1 = DateUtils.parse_date_flexible(date1_str)
        date2 = DateUtils.parse_date_flexible(date2_str)
        
        if date1 and date2:
            return (date2 - date1).days
        return None

class ResponseFormatter:
    """Format responses according to requirements"""
    
    @staticmethod
    def format_answer_array(answers: List[Any]) -> List[Any]:
        """Format answers as array for questions like the movie example"""
        formatted = []
        for answer in answers:
            if isinstance(answer, float):
                if answer.is_integer():
                    formatted.append(int(answer))
                else:
                    formatted.append(round(answer, 6))
            else:
                formatted.append(answer)
        return formatted
    
    @staticmethod
    def format_answer_dict(answers: Dict[str, Any]) -> Dict[str, Any]:
        """Format answers as dictionary for structured responses"""
        formatted = {}
        for key, value in answers.items():
            if isinstance(value, float):
                if value.is_integer():
                    formatted[key] = int(value)
                else:
                    formatted[key] = round(value, 6)
            else:
                formatted[key] = value
        return formatted

class ValidationUtils:
    """Validation utilities"""
    
    @staticmethod
    def validate_api_response(response_data: Any) -> bool:
        """Validate that response matches expected format"""
        if isinstance(response_data, list):
            return len(response_data) > 0
        elif isinstance(response_data, dict):
            return len(response_data) > 0
        return False
    
    @staticmethod
    def validate_base64_image(base64_str: str, max_size_kb: int = 100) -> bool:
        """Validate base64 encoded image"""
        try:
            if not base64_str.startswith('data:image/'):
                return False
            
            # Extract base64 part
            base64_data = base64_str.split(',')[1]
            image_data = base64.b64decode(base64_data)
            
            # Check size
            size_kb = len(image_data) / 1024
            return size_kb <= max_size_kb
            
        except:
            return False

# Environment variable helpers
def get_env_var(var_name: str, default: str = None) -> str:
    """Get environment variable with optional default"""
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is required")
    return value

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log', mode='a')
        ]
    )