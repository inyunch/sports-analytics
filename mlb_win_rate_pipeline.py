"""
Complete MLB Win Rate Prediction Pipeline
Uses Pythagorean expectation approximated from WAR to predict team win rates.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.mlb_data_loader import load_mlb_data
from src.data_processing.war_feature_engineering import WARFeatureEngineer
from src.analytics.pythagorean_eda import PythagoreanEDA
from src.analytics.win_rate_models import WinRatePredictor


def main():
    """
    Main pipeline execution.
    """
    print("=" * 80)
    print(" " * 20 + "MLB Win Rate Prediction Pipeline")
    print(" " * 15 + "Pythagorean Expectation from WAR Model")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Load and Clean Data
    # ========================================================================
    print("\n[STEP 1/5] Loading MLB Data from Raw Folder...")
    print("-" * 80)

    batters, pitchers, standings = load_mlb_data(
        data_dir="data/raw",
        min_pa=100,
        min_ip=20.0
    )

    print(f"\nData Summary:")
    print(f"  • Batters: {len(batters):,} players across {batters['Season'].nunique()} seasons")
    print(f"  • Pitchers: {len(pitchers):,} players across {pitchers['Season'].nunique()} seasons")
    print(f"  • Team-Seasons: {len(standings):,}")
    print(f"  • Seasons: {sorted(batters['Season'].unique())}")

    # ========================================================================
    # STEP 2: Feature Engineering
    # ========================================================================
    print("\n[STEP 2/5] Engineering Features from WAR...")
    print("-" * 80)

    # Initialize feature engineer
    engineer = WARFeatureEngineer(
        k=1.83,  # Pythagorean exponent
        runs_per_win=10.0,
        replacement_level_runs=4.1,
        pa_per_position=600
    )

    # Create team-season features
    team_features = engineer.create_team_season_features(batters, pitchers)

    # Add derived features
    team_features = engineer.add_derived_features(team_features)

    print(f"\nFeatures created for {len(team_features)} team-seasons")
    print(f"\nSample features:")
    print(team_features[['Team', 'Season', 'Offensive_WAR', 'Pitching_WAR',
                         'RS', 'RA', 'Pythagorean_Expectation', 'Expected_Wins']].head(10))

    # Save features
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    team_features.to_csv(output_dir / "team_features_pythagorean.csv", index=False)
    print(f"\n[OK] Features saved to: {output_dir / 'team_features_pythagorean.csv'}")

    # ========================================================================
    # STEP 3: Exploratory Data Analysis
    # ========================================================================
    print("\n[STEP 3/5] Performing Exploratory Data Analysis...")
    print("-" * 80)

    # Initialize EDA analyzer
    eda = PythagoreanEDA(team_features)

    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summaries = eda.generate_summary_statistics()

    print("\n--- Overall Statistics ---")
    print(summaries['overall'].round(2))

    print("\n--- By Season Statistics ---")
    print(summaries['by_season'].round(2))

    if 'top_teams' in summaries:
        print("\n--- Top 10 Teams by Expected Wins ---")
        print(summaries['top_teams'].to_string(index=False))

    # Create EDA report with visualizations
    print("\nCreating visualizations...")
    eda_output = Path("outputs/eda")
    eda.create_eda_report(output_dir=str(eda_output))

    # ========================================================================
    # STEP 4: Predictive Modeling (if actual results available)
    # ========================================================================
    print("\n[STEP 4/5] Building Predictive Models...")
    print("-" * 80)

    # Check if we have actual win rates
    # NOTE: Since we're working with raw data only, we don't have actual win-loss records
    # We'll use estimated wins from standings as a proxy for demonstration
    if 'Estimated_Wins' in standings.columns:
        print("\nNote: Using estimated wins from WAR as proxy for demonstration.")
        print("For real predictions, actual win-loss records are needed.")

        # Merge standings data
        team_features_with_actuals = team_features.merge(
            standings[['Team', 'Season', 'Estimated_Wins', 'Estimated_Win_Rate']],
            on=['Team', 'Season'],
            how='left'
        )

        # Rename for modeling
        team_features_with_actuals['Actual_Win_Rate'] = team_features_with_actuals['Estimated_Win_Rate']

        # Initialize predictor
        predictor = WinRatePredictor(random_state=42)

        # Compare models (use 2025 as test season if available, otherwise 2024)
        available_seasons = sorted(team_features_with_actuals['Season'].unique())
        test_seasons = [available_seasons[-1]] if len(available_seasons) > 1 else None

        if test_seasons and len(available_seasons) > 1:
            try:
                comparison_df = predictor.compare_models(
                    team_features_with_actuals,
                    target_col='Actual_Win_Rate',
                    test_seasons=test_seasons
                )

                # Visualize results
                model_output = Path("outputs/models")
                model_output.mkdir(parents=True, exist_ok=True)

                print("\nCreating model comparison visualizations...")
                predictor.plot_model_comparison(save_path=model_output / "model_comparison.png")

                # Feature importance for best model
                best_model = comparison_df.iloc[0]['Model']
                print(f"\nPlotting feature importance for best model: {best_model}")
                try:
                    predictor.plot_feature_importance(
                        model_name=best_model,
                        save_path=model_output / "feature_importance.png"
                    )
                except:
                    print(f"  (Feature importance not available for {best_model})")

                # Save models
                predictor.save_models(output_dir="models")

                print(f"\n[OK] Models and visualizations saved to: {model_output}")
            except Exception as e:
                print(f"\nSkipping modeling due to insufficient data: {str(e)}")
        else:
            print("\nSkipping modeling: Need multiple seasons for train/test split")
    else:
        print("\nSkipping modeling: Actual win-loss records not available")
        print("Models can be trained once actual results data is integrated.")

    # ========================================================================
    # STEP 5: Summary and Next Steps
    # ========================================================================
    print("\n[STEP 5/5] Pipeline Summary")
    print("=" * 80)

    print("\n[COMPLETE] Pipeline Complete!")
    print("\nOutputs generated:")
    print(f"  1. Processed features: data/processed/team_features_pythagorean.csv")
    print(f"  2. EDA visualizations: outputs/eda/")
    print(f"  3. Model results: outputs/models/ (if applicable)")
    print(f"  4. Trained models: models/ (if applicable)")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("  1. Review EDA visualizations in outputs/eda/")
    print("  2. Examine correlation between WAR components and expected win rate")
    print("  3. Launch Streamlit app: streamlit run streamlit_win_rate_app.py")
    print("  4. For production: integrate actual win-loss records for validation")
    print("=" * 80)

    print("\n>> Key Findings:")
    print("-" * 80)

    # Calculate some interesting statistics
    if not team_features.empty:
        # Best and worst teams by Pythagorean expectation
        best_team = team_features.nlargest(1, 'Pythagorean_Expectation').iloc[0]
        worst_team = team_features.nsmallest(1, 'Pythagorean_Expectation').iloc[0]

        print(f"\n  Best Expected Win Rate:")
        print(f"    • {best_team['Team']} ({int(best_team['Season'])})")
        print(f"    • Pythagorean Expectation: {best_team['Pythagorean_Expectation']:.3f}")
        print(f"    • Expected Wins: {best_team['Expected_Wins']:.1f}")
        print(f"    • Offensive WAR: {best_team['Offensive_WAR']:.1f} | Pitching WAR: {best_team['Pitching_WAR']:.1f}")

        print(f"\n  Worst Expected Win Rate:")
        print(f"    • {worst_team['Team']} ({int(worst_team['Season'])})")
        print(f"    • Pythagorean Expectation: {worst_team['Pythagorean_Expectation']:.3f}")
        print(f"    • Expected Wins: {worst_team['Expected_Wins']:.1f}")
        print(f"    • Offensive WAR: {worst_team['Offensive_WAR']:.1f} | Pitching WAR: {worst_team['Pitching_WAR']:.1f}")

        # Average values
        avg_off_war = team_features['Offensive_WAR'].mean()
        avg_pit_war = team_features['Pitching_WAR'].mean()
        avg_pe = team_features['Pythagorean_Expectation'].mean()

        print(f"\n  League Averages:")
        print(f"    • Offensive WAR: {avg_off_war:.1f}")
        print(f"    • Pitching WAR: {avg_pit_war:.1f}")
        print(f"    • Pythagorean Expectation: {avg_pe:.3f} ({avg_pe * 162:.1f} wins)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
