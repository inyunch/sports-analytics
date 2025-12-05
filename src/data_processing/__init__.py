"""
Data processing module for baseball analytics.
Handles data loading, cleaning, and integration.
"""

from .mlb_data_loader import load_mlb_data, MLBDataLoader
from .war_feature_engineering import WARFeatureEngineer, extract_season_from_data

__all__ = [
    'load_mlb_data',
    'MLBDataLoader',
    'WARFeatureEngineer',
    'extract_season_from_data'
]
