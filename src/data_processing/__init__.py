"""
Data processing module for baseball analytics.
Handles data loading, cleaning, and integration.
"""

from .data_loader import load_batter_data, load_pitcher_data
from .data_cleaner import clean_batter_data, clean_pitcher_data
from .data_integration import integrate_datasets

__all__ = [
    'load_batter_data',
    'load_pitcher_data',
    'clean_batter_data',
    'clean_pitcher_data',
    'integrate_datasets'
]
