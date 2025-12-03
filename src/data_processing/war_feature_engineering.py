"""
WAR-based feature engineering for Pythagorean expectation model.
Approximates runs scored (RS) and runs allowed (RA) from player WAR.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class WARFeatureEngineer:
    """
    Transforms player-level WAR statistics into team-level features
    for Pythagorean win expectation modeling.
    """

    def __init__(self,
                 k: float = 1.83,
                 runs_per_win: float = 10.0,
                 replacement_level_runs: float = 4.1,
                 pa_per_position: int = 600):
        """
        Initialize feature engineer with model constants.

        Args:
            k: Pythagorean exponent (typically 1.83 for baseball)
            runs_per_win: Runs per win conversion factor
            replacement_level_runs: Replacement level runs per 600 PA
            pa_per_position: Plate appearances per position for replacement
        """
        self.k = k
        self.runs_per_win = runs_per_win
        self.replacement_level_runs = replacement_level_runs
        self.pa_per_position = pa_per_position

    def create_team_season_features(self,
                                    batters: pd.DataFrame,
                                    pitchers: pd.DataFrame) -> pd.DataFrame:
        """
        Create team-season level features from player data.

        Args:
            batters: DataFrame with batter statistics (must have 'Team', 'Season', 'WAR')
            pitchers: DataFrame with pitcher statistics (must have 'Team', 'Season', 'WAR')

        Returns:
            DataFrame with team-season features including RS, RA, and Pythagorean expectation
        """
        # Aggregate offensive WAR by team-season
        batter_war = batters.groupby(['Team', 'Season'])['WAR'].sum().reset_index()
        batter_war.columns = ['Team', 'Season', 'Offensive_WAR']

        # Aggregate pitching WAR by team-season
        pitcher_war = pitchers.groupby(['Team', 'Season'])['WAR'].sum().reset_index()
        pitcher_war.columns = ['Team', 'Season', 'Pitching_WAR']

        # Merge offensive and pitching WAR
        team_features = batter_war.merge(pitcher_war, on=['Team', 'Season'], how='outer')
        team_features.fillna(0, inplace=True)

        # Calculate runs scored (RS) approximation
        team_features['RS'] = self._calculate_runs_scored(team_features['Offensive_WAR'])

        # Calculate runs allowed (RA) approximation
        team_features['RA'] = self._calculate_runs_allowed(team_features['Pitching_WAR'])

        # Calculate Pythagorean expectation
        team_features['Pythagorean_Expectation'] = self._calculate_pythagorean_expectation(
            team_features['RS'],
            team_features['RA']
        )

        # Add run differential
        team_features['Run_Differential'] = team_features['RS'] - team_features['RA']

        # Add total WAR
        team_features['Total_WAR'] = team_features['Offensive_WAR'] + team_features['Pitching_WAR']

        return team_features

    def _calculate_runs_scored(self, offensive_war: pd.Series) -> pd.Series:
        """
        Approximate runs scored from offensive WAR.

        Formula: RS = (replacement level runs for 9 positions) + (Offensive WAR * runs_per_win)

        Args:
            offensive_war: Series of team offensive WAR

        Returns:
            Series of estimated runs scored
        """
        # Replacement level runs for 9 lineup spots (assuming 162 games, ~600 PA per position)
        # 4.1 runs per 600 PA * 9 positions = 36.9 runs per position per season
        # Scaled by (162 games * 4 PA/game/position) / 600 PA
        replacement_runs = 9 * self.replacement_level_runs * (162 * 4.5) / self.pa_per_position

        # Add runs from WAR above replacement
        # WAR is already in wins, convert to runs
        war_runs = offensive_war * self.runs_per_win

        return replacement_runs + war_runs

    def _calculate_runs_allowed(self, pitching_war: pd.Series) -> pd.Series:
        """
        Approximate runs allowed from pitching WAR.

        Formula: RA = (league average RA) - (Pitching WAR * runs_per_win)

        Args:
            pitching_war: Series of team pitching WAR

        Returns:
            Series of estimated runs allowed
        """
        # Baseline runs allowed (league average, approximately 4.5 runs per game * 162 games)
        baseline_ra = 4.5 * 162

        # Subtract runs prevented by pitching WAR
        war_runs_prevented = pitching_war * self.runs_per_win

        return baseline_ra - war_runs_prevented

    def _calculate_pythagorean_expectation(self, rs: pd.Series, ra: pd.Series) -> pd.Series:
        """
        Calculate Pythagorean win expectation.

        Formula: PE = RS^k / (RS^k + RA^k)

        Args:
            rs: Runs scored
            ra: Runs allowed

        Returns:
            Pythagorean win expectation (0 to 1)
        """
        # Avoid division by zero
        rs = rs.clip(lower=1)
        ra = ra.clip(lower=1)

        rs_k = np.power(rs, self.k)
        ra_k = np.power(ra, self.k)

        pythagorean = rs_k / (rs_k + ra_k)

        return pythagorean

    def calculate_matchup_win_probability(self,
                                         pe_team_a: float,
                                         pe_team_b: float) -> float:
        """
        Calculate win probability for team A vs team B based on Pythagorean expectations.

        Formula: win_prob_A = PE_A / (PE_A + PE_B)

        Args:
            pe_team_a: Pythagorean expectation for team A
            pe_team_b: Pythagorean expectation for team B

        Returns:
            Win probability for team A (0 to 1)
        """
        if pe_team_a + pe_team_b == 0:
            return 0.5

        return pe_team_a / (pe_team_a + pe_team_b)

    def add_derived_features(self, team_features: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional derived features for modeling.

        Args:
            team_features: DataFrame with base team features

        Returns:
            DataFrame with additional features
        """
        df = team_features.copy()

        # WAR ratios
        df['WAR_Balance'] = df['Offensive_WAR'] / (df['Pitching_WAR'] + 0.1)  # Avoid division by zero

        # Runs per game
        df['RS_per_game'] = df['RS'] / 162
        df['RA_per_game'] = df['RA'] / 162

        # Expected wins (162 games)
        df['Expected_Wins'] = df['Pythagorean_Expectation'] * 162

        # Strength metrics
        df['Offensive_Strength'] = df['Offensive_WAR'] / df['Total_WAR'].clip(lower=0.1)
        df['Pitching_Strength'] = df['Pitching_WAR'] / df['Total_WAR'].clip(lower=0.1)

        return df

    def prepare_modeling_dataset(self,
                                batters: pd.DataFrame,
                                pitchers: pd.DataFrame,
                                actual_results: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare complete dataset for modeling with all features.

        Args:
            batters: Batter statistics DataFrame
            pitchers: Pitcher statistics DataFrame
            actual_results: Optional DataFrame with actual wins/losses/games
                          (must have 'Team', 'Season', 'Wins', 'Losses', 'Games')

        Returns:
            Complete modeling dataset with features and target (if provided)
        """
        # Create base features
        team_features = self.create_team_season_features(batters, pitchers)

        # Add derived features
        team_features = self.add_derived_features(team_features)

        # Merge with actual results if provided
        if actual_results is not None:
            team_features = team_features.merge(
                actual_results[['Team', 'Season', 'Wins', 'Losses', 'Games']],
                on=['Team', 'Season'],
                how='left'
            )

            # Calculate actual win rate
            team_features['Actual_Win_Rate'] = (
                team_features['Wins'] / team_features['Games']
            )

            # Calculate residual (actual - expected)
            team_features['Win_Rate_Residual'] = (
                team_features['Actual_Win_Rate'] - team_features['Pythagorean_Expectation']
            )

        return team_features


def extract_season_from_data(batters: pd.DataFrame,
                             pitchers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract season information from player data if not already present.

    Args:
        batters: Batter DataFrame
        pitchers: Pitcher DataFrame

    Returns:
        Tuple of (batters_with_season, pitchers_with_season)
    """
    # If Season column doesn't exist, try to infer it
    if 'Season' not in batters.columns:
        # Add a default season or extract from filename/path
        batters['Season'] = 2024  # Default for now

    if 'Season' not in pitchers.columns:
        pitchers['Season'] = 2024  # Default for now

    return batters, pitchers


if __name__ == "__main__":
    # Test the feature engineering
    print("WAR Feature Engineering Module")
    print("=" * 60)

    # Create sample data
    sample_batters = pd.DataFrame({
        'Team': ['NYY', 'NYY', 'LAD', 'LAD'],
        'Season': [2024, 2024, 2024, 2024],
        'WAR': [8.0, 6.5, 9.0, 7.5]
    })

    sample_pitchers = pd.DataFrame({
        'Team': ['NYY', 'NYY', 'LAD', 'LAD'],
        'Season': [2024, 2024, 2024, 2024],
        'WAR': [5.0, 4.5, 6.0, 5.5]
    })

    # Initialize feature engineer
    engineer = WARFeatureEngineer()

    # Create features
    features = engineer.create_team_season_features(sample_batters, sample_pitchers)

    print("\nTeam-Season Features:")
    print(features)

    print("\n" + "=" * 60)
    print("Feature engineering complete!")
