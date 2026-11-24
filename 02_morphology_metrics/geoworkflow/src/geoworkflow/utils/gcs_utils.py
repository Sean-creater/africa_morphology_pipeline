"""
Google Cloud Storage utilities for Open Buildings data access.
"""
from typing import Optional, Union, List, Any
from pathlib import Path
import gcsfs
import logging
import pandas as pd


class GCSClient:
    """
    Wrapper for GCS access with multiple auth methods.
    
    Supports:
    - Anonymous access (for public datasets like Open Buildings)
    - Service account authentication
    - Default credentials
    """
    
    def __init__(
        self,
        service_account_key: Optional[Path] = None,
        use_anonymous: bool = True,
        cache_timeout: int = 300,  # Cache directory listings for 5 minutes
        block_size: int = 5 * 2**20  # 5MB blocks for faster streaming
    ):
        """
        Initialize GCS client with performance optimizations.

        Args:
            service_account_key: Path to service account JSON key file
            use_anonymous: Use anonymous access (for public data)
            cache_timeout: Cache timeout in seconds for file metadata
            block_size: Block size for streaming reads (larger = fewer requests)

        Raises:
            ImportError: If gcsfs is not installed
            ValueError: If authentication fails
        """
        self.logger = logging.getLogger(__name__)

        # Initialize gcsfs filesystem with performance settings
        try:
            fs_kwargs = {
                'cache_timeout': cache_timeout,
                'block_size': block_size,
            }

            if use_anonymous:
                self.fs = gcsfs.GCSFileSystem(token='anon', **fs_kwargs)
                self.logger.info("Initialized GCS client with anonymous access (optimized)")
            elif service_account_key:
                self.fs = gcsfs.GCSFileSystem(token=str(service_account_key), **fs_kwargs)
                self.logger.info(f"Initialized GCS client with service account: {service_account_key}")
            else:
                self.fs = gcsfs.GCSFileSystem(**fs_kwargs)
                self.logger.info("Initialized GCS client with default credentials")

        except Exception as e:
            raise ValueError(f"Failed to initialize GCS client: {e}")
    
    def file_exists(self, path: str) -> bool:
        """
        Check if GCS file exists.
        
        Args:
            path: GCS path (e.g., 'gs://bucket/path/file.csv.gz')
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            # Remove gs:// prefix if present
            path = path.replace('gs://', '')
            return self.fs.exists(path)
        except Exception as e:
            self.logger.warning(f"Error checking if file exists {path}: {e}")
            return False
    
    def read_csv_gz(
        self,
        path: str,
        usecols: Optional[List[str]] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Read compressed CSV from GCS with optimizations.

        Args:
            path: GCS path to CSV.gz file
            usecols: List of columns to read (reduces memory and parsing time)
            **kwargs: Additional arguments passed to pd.read_csv

        Returns:
            DataFrame with CSV contents

        Example:
            >>> client = GCSClient(use_anonymous=True)
            >>> df = client.read_csv_gz('gs://bucket/data.csv.gz')
        """
        # Remove gs:// prefix if present
        path = path.replace('gs://', '')

        # Default column names for Open Buildings v3 polygons (no header files)
        default_names = ['latitude', 'longitude', 'area_in_meters', 'confidence', 'geometry', 'full_plus_code']

        # Allow override via kwargs
        if 'names' not in kwargs:
            kwargs['names'] = default_names
        if 'header' not in kwargs:
            kwargs['header'] = None  # No header row

        # Add column selection if specified
        if usecols:
            kwargs['usecols'] = usecols

        # Use faster C engine and optimize dtypes
        if 'engine' not in kwargs:
            kwargs['engine'] = 'c'  # C engine is faster than python

        # Pre-define dtypes for faster parsing
        if 'dtype' not in kwargs:
            kwargs['dtype'] = {
                'latitude': 'float32',  # Use float32 instead of float64 to save memory
                'longitude': 'float32',
                'area_in_meters': 'float32',
                'confidence': 'float32'
            }

        with self.fs.open(path, 'rb') as f:
            return pd.read_csv(f, compression='gzip', **kwargs)
    
    def list_files(self, prefix: str, suffix: str = '') -> List[str]:
        """
        List files with given prefix.
        
        Args:
            prefix: GCS path prefix (e.g., 'gs://bucket/path/')
            suffix: Optional file suffix filter (e.g., '.csv.gz')
            
        Returns:
            List of GCS paths matching prefix and suffix
            
        Example:
            >>> client = GCSClient(use_anonymous=True)
            >>> files = client.list_files('gs://open-buildings-data/v3/', suffix='.csv.gz')
        """
        # Remove gs:// prefix if present
        prefix = prefix.replace('gs://', '')
        
        try:
            files = self.fs.ls(prefix)
            
            if suffix:
                files = [f for f in files if f.endswith(suffix)]
            
            # Add gs:// prefix back
            return [f'gs://{f}' for f in files]
            
        except Exception as e:
            self.logger.error(f"Error listing files with prefix {prefix}: {e}")
            return []
    
    def download_file(self, gcs_path: str, local_path: Path) -> bool:
        """
        Download file from GCS to local path.
        
        Args:
            gcs_path: GCS path (e.g., 'gs://bucket/file.csv.gz')
            local_path: Local destination path
            
        Returns:
            True if successful, False otherwise
        """
        # Remove gs:// prefix if present
        gcs_path = gcs_path.replace('gs://', '')
        
        try:
            self.fs.get(gcs_path, str(local_path))
            return True
        except Exception as e:
            self.logger.error(f"Error downloading {gcs_path}: {e}")
            return False
    
    def get_file_size(self, path: str) -> Optional[int]:
        """
        Get file size in bytes.
        
        Args:
            path: GCS path
            
        Returns:
            File size in bytes, or None if error
        """
        # Remove gs:// prefix if present
        path = path.replace('gs://', '')
        
        try:
            info = self.fs.info(path)
            return info.get('size')
        except Exception as e:
            self.logger.warning(f"Error getting file size for {path}: {e}")
            return None
