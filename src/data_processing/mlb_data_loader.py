"""
MLB Data Loader for multi-season batter and pitcher statistics.
Loads data from the raw folder (2023-2025 seasons).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict


class MLBDataLoader:
    """Load and preprocess MLB batter and pitcher data from raw CSV files."""

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader.

        Args:
            data_dir: Path to directory containing raw CSV files
        """
        self.data_dir = Path(data_dir)

    def load_all_seasons(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all available seasons for batters and pitchers.

        Returns:
            Tuple of (all_batters, all_pitchers) DataFrames
        """
        # Load batters from all seasons
        batter_files = sorted(self.data_dir.glob("MLB_Batter_Stat_*.csv"))
        pitchers_files = sorted(self.data_dir.glob("MLB_Pitcher_Stat_*.csv"))

        print(f"Found {len(batter_files)} batter files and {len(pitchers_files)} pitcher files")

        # Load and combine batters
        batter_dfs = []
        for file_path in batter_files:
            # Extract season from filename
            season = int(file_path.stem.split('_')[-1])
            df = pd.read_csv(file_path)
            df['Season'] = season
            batter_dfs.append(df)
            print(f"  Loaded {len(df)} batters from {season}")

        all_batters = pd.concat(batter_dfs, ignore_index=True) if batter_dfs else pd.DataFrame()

        # Load and combine pitchers
        pitcher_dfs = []
        for file_path in pitchers_files:
            # Extract season from filename
            season = int(file_path.stem.split('_')[-1])
            df = pd.read_csv(file_path)
            df['Season'] = season
            pitcher_dfs.append(df)
            print(f"  Loaded {len(df)} pitchers from {season}")

        all_pitchers = pd.concat(pitcher_dfs, ignore_index=True) if pitcher_dfs else pd.DataFrame()

        return all_batters, all_pitchers

    def clean_batter_data(self, batters: pd.DataFrame, min_pa: int = 100) -> pd.DataFrame:
        """
        Clean and preprocess batter data.

        Args:
            batters: Raw batter DataFrame
            min_pa: Minimum plate appearances to include player

        Returns:
            Cleaned batter DataFrame
        """
        if batters.empty:
            return batters

        df = batters.copy()

        # Filter by minimum PA
        if 'PA' in df.columns:
            df = df[df['PA'] >= min_pa]

        # Handle missing WAR values
        if 'WAR' in df.columns:
            df['WAR'] = df['WAR'].fillna(0)
            # Convert to numeric, handling any non-numeric values
            df['WAR'] = pd.to_numeric(df['WAR'], errors='coerce').fillna(0)

        # Clean team names (remove special characters)
        if 'Team' in df.columns:
            df['Team'] = df['Team'].astype(str).str.strip()
            # Handle players who played for multiple teams (keep total stats)
            # Players with "2TM" or similar indicators have multiple rows
            # We'll aggregate by player-season keeping the row with most PA
            if 'Player' in df.columns and 'PA' in df.columns:
                # For players with multiple teams, keep the combined stats row if it exists
                # or sum their stats across teams
                df = self._handle_multi_team_players(df, stat_type='batter')

        # Ensure numeric columns are properly typed
        numeric_cols = ['WAR', 'PA', 'AB', 'R', 'H', 'HR', 'RBI', 'SB', 'BB', 'SO',
                       'BA', 'OBP', 'SLG', 'OPS', 'OPS+']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Add league column if not present
        if 'Lg' in df.columns and 'League' not in df.columns:
            df['League'] = df['Lg']

        print(f"Cleaned batter data: {len(df)} players after filtering (min PA={min_pa})")

        return df

    def clean_pitcher_data(self, pitchers: pd.DataFrame, min_ip: float = 20.0) -> pd.DataFrame:
        """
        Clean and preprocess pitcher data.

        Args:
            pitchers: Raw pitcher DataFrame
            min_ip: Minimum innings pitched to include pitcher

        Returns:
            Cleaned pitcher DataFrame
        """
        if pitchers.empty:
            return pitchers

        df = pitchers.copy()

        # Filter by minimum IP
        if 'IP' in df.columns:
            # Convert IP to numeric (handle fractional innings like "192.1")
            df['IP'] = df['IP'].apply(self._convert_innings_pitched)
            df = df[df['IP'] >= min_ip]

        # Handle missing WAR values
        if 'WAR' in df.columns:
            df['WAR'] = df['WAR'].fillna(0)
            df['WAR'] = pd.to_numeric(df['WAR'], errors='coerce').fillna(0)

        # Clean team names
        if 'Team' in df.columns:
            df['Team'] = df['Team'].astype(str).str.strip()
            # Handle multi-team pitchers
            if 'Player' in df.columns and 'IP' in df.columns:
                df = self._handle_multi_team_players(df, stat_type='pitcher')

        # Ensure numeric columns are properly typed
        numeric_cols = ['WAR', 'W', 'L', 'ERA', 'G', 'GS', 'IP', 'SO', 'BB', 'HR',
                       'WHIP', 'FIP', 'ERA+']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Add league column if not present
        if 'Lg' in df.columns and 'League' not in df.columns:
            df['League'] = df['Lg']

        print(f"Cleaned pitcher data: {len(df)} pitchers after filtering (min IP={min_ip})")

        return df

    def _convert_innings_pitched(self, ip_value) -> float:
        """
        Convert innings pitched string to decimal.
        Example: "192.1" (192 1/3 innings) -> 192.333...

        Args:
            ip_value: Innings pitched value (string or numeric)

        Returns:
            Decimal innings pitched
        """
        try:
            if pd.isna(ip_value):
                return 0.0

            ip_str = str(ip_value)

            if '.' in ip_str:
                whole, partial = ip_str.split('.')
                # Partial inning: .1 = 1/3, .2 = 2/3
                return float(whole) + int(partial) / 3.0
            else:
                return float(ip_str)
        except:
            return 0.0

    def _handle_multi_team_players(self, df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
        """
        Handle players who played for multiple teams in a season.

        Strategy: Keep only the aggregated "total" row (2TM, 3TM, etc.) if it exists,
        otherwise sum stats across teams.

        Args:
            df: Player DataFrame
            stat_type: 'batter' or 'pitcher'

        Returns:
            DataFrame with one row per player-season
        """
        if 'Player' not in df.columns or 'Season' not in df.columns:
            return df

        # Identify multi-team indicators
        multi_team_indicators = ['2TM', '3TM', 'TOT']

        result_rows = []

        for (player, season), group in df.groupby(['Player', 'Season']):
            if len(group) == 1:
                # Player only on one team, keep as is
                result_rows.append(group.iloc[0])
            else:
                # Player on multiple teams
                # Check if there's a total row
                total_row = group[group['Team'].isin(multi_team_indicators)]

                if not total_row.empty:
                    # Use the total row
                    result_rows.append(total_row.iloc[0])
                else:
                    # No total row, create one by summing stats
                    combined = group.iloc[0].copy()
                    combined['Team'] = '2TM'  # Mark as multi-team

                    # Sum counting stats
                    if stat_type == 'batter':
                        sum_cols = ['PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI',
                                   'SB', 'CS', 'BB', 'SO', 'GIDP', 'HBP', 'SH', 'SF', 'IBB']
                    else:  # pitcher
                        sum_cols = ['W', 'L', 'G', 'GS', 'IP', 'R', 'ER', 'HR', 'BB',
                                   'SO', 'HBP', 'BK', 'WP', 'BF']

                    for col in sum_cols:
                        if col in group.columns:
                            combined[col] = group[col].sum()

                    # Recalculate rate stats for batters
                    if stat_type == 'batter' and 'AB' in combined and combined['AB'] > 0:
                        if 'H' in combined:
                            combined['BA'] = combined['H'] / combined['AB']
                        if 'PA' in combined and combined['PA'] > 0:
                            combined['OBP'] = (combined['H'] + combined['BB'] + combined['HBP']) / combined['PA']

                    # WAR typically provided as total
                    if 'WAR' in group.columns:
                        combined['WAR'] = group['WAR'].sum()

                    result_rows.append(combined)

        return pd.DataFrame(result_rows)

    def create_team_standings(self, batters: pd.DataFrame, pitchers: pd.DataFrame) -> pd.DataFrame:
        """
        Create estimated team standings based on WAR.

        Note: This is an approximation since we don't have actual win-loss records.
        We'll estimate based on total team WAR.

        Args:
            batters: Cleaned batter DataFrame
            pitchers: Cleaned pitcher DataFrame

        Returns:
            DataFrame with estimated team standings by season
        """
        # Aggregate WAR by team and season
        batter_war = batters.groupby(['Team', 'Season'])['WAR'].sum().reset_index()
        batter_war.columns = ['Team', 'Season', 'Offensive_WAR']

        pitcher_war = pitchers.groupby(['Team', 'Season'])['WAR'].sum().reset_index()
        pitcher_war.columns = ['Team', 'Season', 'Pitching_WAR']

        # Merge
        standings = batter_war.merge(pitcher_war, on=['Team', 'Season'], how='outer')
        standings.fillna(0, inplace=True)

        # Calculate total WAR
        standings['Total_WAR'] = standings['Offensive_WAR'] + standings['Pitching_WAR']

        # Estimate wins (baseline 81 wins + WAR)
        standings['Estimated_Wins'] = 81 + standings['Total_WAR']
        standings['Estimated_Losses'] = 162 - standings['Estimated_Wins']
        standings['Games'] = 162

        # Calculate estimated win rate
        standings['Estimated_Win_Rate'] = standings['Estimated_Wins'] / standings['Games']

        return standings.sort_values(['Season', 'Estimated_Wins'], ascending=[True, False])


def load_mlb_data(data_dir: str = "data/raw",
                 min_pa: int = 100,
                 min_ip: float = 20.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and clean all MLB data.

    Args:
        data_dir: Path to raw data directory
        min_pa: Minimum plate appearances for batters
        min_ip: Minimum innings pitched for pitchers

    Returns:
        Tuple of (batters, pitchers, team_standings)
    """
    loader = MLBDataLoader(data_dir)

    # Load all seasons
    batters, pitchers = loader.load_all_seasons()

    # Clean data
    batters = loader.clean_batter_data(batters, min_pa=min_pa)
    pitchers = loader.clean_pitcher_data(pitchers, min_ip=min_ip)

    # Create team standings
    standings = loader.create_team_standings(batters, pitchers)

    return batters, pitchers, standings


if __name__ == "__main__":
    print("=" * 60)
    print("MLB Data Loader - Testing")
    print("=" * 60)

    # Load data
    batters, pitchers, standings = load_mlb_data()

    print(f"\nTotal batters loaded: {len(batters)}")
    print(f"Total pitchers loaded: {len(pitchers)}")
    print(f"Total team-seasons: {len(standings)}")

    print("\nSample batter data:")
    print(batters[['Player', 'Team', 'Season', 'WAR', 'PA', 'OPS']].head())

    print("\nSample pitcher data:")
    print(pitchers[['Player', 'Team', 'Season', 'WAR', 'IP', 'ERA']].head())

    print("\nTop 5 teams by Total WAR (2024):")
    print(standings[standings['Season'] == 2024].head()[
        ['Team', 'Season', 'Offensive_WAR', 'Pitching_WAR', 'Total_WAR', 'Estimated_Wins']
    ])

    print("\n" + "=" * 60)
