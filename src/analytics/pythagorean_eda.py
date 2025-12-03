"""
Exploratory Data Analysis for Pythagorean Expectation Model.
Visualizations and statistical analysis of WAR-based win rate predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class PythagoreanEDA:
    """
    Exploratory data analysis tools for Pythagorean expectation modeling.
    """

    def __init__(self, team_features: pd.DataFrame, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize EDA analyzer.

        Args:
            team_features: DataFrame with team-season features
            style: Matplotlib style
        """
        self.team_features = team_features
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        # Set color palette
        self.colors = sns.color_palette("husl", 10)

    def plot_war_distributions(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distributions of offensive and pitching WAR.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('WAR Distributions by Season', fontsize=16, fontweight='bold')

        df = self.team_features

        # Offensive WAR distribution
        for season in sorted(df['Season'].unique()):
            season_data = df[df['Season'] == season]['Offensive_WAR']
            axes[0, 0].hist(season_data, alpha=0.5, label=f'{season}', bins=15)

        axes[0, 0].set_xlabel('Offensive WAR', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Offensive WAR Distribution', fontsize=13)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Pitching WAR distribution
        for season in sorted(df['Season'].unique()):
            season_data = df[df['Season'] == season]['Pitching_WAR']
            axes[0, 1].hist(season_data, alpha=0.5, label=f'{season}', bins=15)

        axes[0, 1].set_xlabel('Pitching WAR', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Pitching WAR Distribution', fontsize=13)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Total WAR distribution
        for season in sorted(df['Season'].unique()):
            season_data = df[df['Season'] == season]['Total_WAR']
            axes[1, 0].hist(season_data, alpha=0.5, label=f'{season}', bins=15)

        axes[1, 0].set_xlabel('Total WAR', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Total WAR Distribution', fontsize=13)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # WAR Balance (Offensive/Pitching ratio)
        if 'WAR_Balance' in df.columns:
            for season in sorted(df['Season'].unique()):
                season_data = df[df['Season'] == season]['WAR_Balance']
                axes[1, 1].hist(season_data, alpha=0.5, label=f'{season}', bins=15)

            axes[1, 1].set_xlabel('WAR Balance (Offensive/Pitching)', fontsize=12)
            axes[1, 1].set_ylabel('Frequency', fontsize=12)
            axes[1, 1].set_title('Team Balance Distribution', fontsize=13)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_pythagorean_scatter(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Pythagorean expectation vs actual win rate (if available).

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Pythagorean Expectation Analysis', fontsize=16, fontweight='bold')

        df = self.team_features

        # If actual win rate available, plot comparison
        if 'Actual_Win_Rate' in df.columns:
            seasons = sorted(df['Season'].unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(seasons)))

            for i, season in enumerate(seasons):
                season_data = df[df['Season'] == season]
                axes[0].scatter(season_data['Pythagorean_Expectation'],
                              season_data['Actual_Win_Rate'],
                              alpha=0.6, s=100, c=[colors[i]], label=f'{season}')

            # Add diagonal line (perfect prediction)
            min_val = min(df['Pythagorean_Expectation'].min(), df['Actual_Win_Rate'].min())
            max_val = max(df['Pythagorean_Expectation'].max(), df['Actual_Win_Rate'].max())
            axes[0].plot([min_val, max_val], [min_val, max_val],
                        'r--', linewidth=2, label='Perfect Prediction')

            # Calculate and display correlation
            corr = df['Pythagorean_Expectation'].corr(df['Actual_Win_Rate'])
            axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                        transform=axes[0].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            axes[0].set_xlabel('Pythagorean Expectation', fontsize=12)
            axes[0].set_ylabel('Actual Win Rate', fontsize=12)
            axes[0].set_title('Expected vs Actual Win Rate', fontsize=13)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Residuals plot
            if 'Win_Rate_Residual' in df.columns:
                for i, season in enumerate(seasons):
                    season_data = df[df['Season'] == season]
                    axes[1].scatter(season_data['Pythagorean_Expectation'],
                                  season_data['Win_Rate_Residual'],
                                  alpha=0.6, s=100, c=[colors[i]], label=f'{season}')

                axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
                axes[1].set_xlabel('Pythagorean Expectation', fontsize=12)
                axes[1].set_ylabel('Residual (Actual - Expected)', fontsize=12)
                axes[1].set_title('Prediction Residuals', fontsize=13)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        else:
            # Just plot distribution of Pythagorean expectations
            for season in sorted(df['Season'].unique()):
                season_data = df[df['Season'] == season]['Pythagorean_Expectation']
                axes[0].hist(season_data, alpha=0.5, label=f'{season}', bins=20)

            axes[0].set_xlabel('Pythagorean Expectation', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].set_title('Pythagorean Expectation Distribution', fontsize=13)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot expected wins
            for season in sorted(df['Season'].unique()):
                season_data = df[df['Season'] == season]['Expected_Wins']
                axes[1].hist(season_data, alpha=0.5, label=f'{season}', bins=20)

            axes[1].set_xlabel('Expected Wins (out of 162)', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].set_title('Expected Wins Distribution', fontsize=13)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_rs_ra_relationship(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot relationship between runs scored and runs allowed.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Runs Scored vs Runs Allowed Analysis', fontsize=16, fontweight='bold')

        df = self.team_features
        seasons = sorted(df['Season'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(seasons)))

        # Scatter plot: RS vs RA
        for i, season in enumerate(seasons):
            season_data = df[df['Season'] == season]
            axes[0].scatter(season_data['RS'], season_data['RA'],
                          alpha=0.6, s=100, c=[colors[i]], label=f'{season}')

        # Add diagonal line (RS = RA, .500 win rate)
        min_val = min(df['RS'].min(), df['RA'].min())
        max_val = max(df['RS'].max(), df['RA'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, alpha=0.5, label='RS = RA (.500)')

        axes[0].set_xlabel('Runs Scored (RS)', fontsize=12)
        axes[0].set_ylabel('Runs Allowed (RA)', fontsize=12)
        axes[0].set_title('Runs Scored vs Runs Allowed', fontsize=13)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Run differential vs Pythagorean expectation
        for i, season in enumerate(seasons):
            season_data = df[df['Season'] == season]
            axes[1].scatter(season_data['Run_Differential'],
                          season_data['Pythagorean_Expectation'],
                          alpha=0.6, s=100, c=[colors[i]], label=f'{season}')

        axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        axes[1].set_xlabel('Run Differential (RS - RA)', fontsize=12)
        axes[1].set_ylabel('Pythagorean Expectation', fontsize=12)
        axes[1].set_title('Run Differential vs Expected Win Rate', fontsize=13)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_correlation_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation heatmap of key features.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Select numeric columns for correlation
        numeric_cols = ['Offensive_WAR', 'Pitching_WAR', 'Total_WAR', 'RS', 'RA',
                       'Run_Differential', 'Pythagorean_Expectation', 'Expected_Wins']

        if 'Actual_Win_Rate' in self.team_features.columns:
            numeric_cols.append('Actual_Win_Rate')

        # Filter to columns that exist
        available_cols = [col for col in numeric_cols if col in self.team_features.columns]

        # Calculate correlation matrix
        corr_matrix = self.team_features[available_cols].corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)

        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def generate_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive summary statistics.

        Returns:
            Dictionary of summary DataFrames
        """
        summaries = {}

        # Overall statistics
        numeric_cols = ['Offensive_WAR', 'Pitching_WAR', 'Total_WAR', 'RS', 'RA',
                       'Pythagorean_Expectation', 'Expected_Wins']
        available_cols = [col for col in numeric_cols if col in self.team_features.columns]

        summaries['overall'] = self.team_features[available_cols].describe()

        # By season statistics
        season_stats = []
        for season in sorted(self.team_features['Season'].unique()):
            season_data = self.team_features[self.team_features['Season'] == season]
            stats = season_data[available_cols].mean()
            stats['Season'] = season
            stats['N_Teams'] = len(season_data)
            season_stats.append(stats)

        summaries['by_season'] = pd.DataFrame(season_stats)

        # Top and bottom teams
        if 'Expected_Wins' in self.team_features.columns:
            summaries['top_teams'] = self.team_features.nlargest(10, 'Expected_Wins')[
                ['Team', 'Season', 'Offensive_WAR', 'Pitching_WAR',
                 'Total_WAR', 'Expected_Wins', 'Pythagorean_Expectation']
            ]

            summaries['bottom_teams'] = self.team_features.nsmallest(10, 'Expected_Wins')[
                ['Team', 'Season', 'Offensive_WAR', 'Pitching_WAR',
                 'Total_WAR', 'Expected_Wins', 'Pythagorean_Expectation']
            ]

        # WAR component analysis
        war_analysis = pd.DataFrame({
            'Metric': ['Offensive WAR', 'Pitching WAR', 'Total WAR'],
            'Mean': [
                self.team_features['Offensive_WAR'].mean(),
                self.team_features['Pitching_WAR'].mean(),
                self.team_features['Total_WAR'].mean()
            ],
            'Std': [
                self.team_features['Offensive_WAR'].std(),
                self.team_features['Pitching_WAR'].std(),
                self.team_features['Total_WAR'].std()
            ],
            'Min': [
                self.team_features['Offensive_WAR'].min(),
                self.team_features['Pitching_WAR'].min(),
                self.team_features['Total_WAR'].min()
            ],
            'Max': [
                self.team_features['Offensive_WAR'].max(),
                self.team_features['Pitching_WAR'].max(),
                self.team_features['Total_WAR'].max()
            ]
        })

        summaries['war_analysis'] = war_analysis

        return summaries

    def create_eda_report(self, output_dir: str = "outputs/eda") -> None:
        """
        Generate complete EDA report with all visualizations and statistics.

        Args:
            output_dir: Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Generating EDA Report...")
        print("=" * 60)

        # Generate plots
        print("\n[1/4] Creating WAR distribution plots...")
        self.plot_war_distributions(save_path=output_path / "war_distributions.png")

        print("[2/4] Creating Pythagorean expectation plots...")
        self.plot_pythagorean_scatter(save_path=output_path / "pythagorean_analysis.png")

        print("[3/4] Creating RS/RA relationship plots...")
        self.plot_rs_ra_relationship(save_path=output_path / "rs_ra_analysis.png")

        print("[4/4] Creating correlation heatmap...")
        self.plot_correlation_heatmap(save_path=output_path / "correlation_heatmap.png")

        # Generate summary statistics
        print("\nGenerating summary statistics...")
        summaries = self.generate_summary_statistics()

        # Save summaries to CSV
        for name, df in summaries.items():
            df.to_csv(output_path / f"summary_{name}.csv", index=True)
            print(f"  Saved: summary_{name}.csv")

        print("\n" + "=" * 60)
        print(f"EDA Report complete! Files saved to: {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    print("Pythagorean EDA Module - Ready for use")
