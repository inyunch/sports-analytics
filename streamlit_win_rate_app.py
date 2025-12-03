"""
Streamlit Web Application for MLB Win Rate Prediction
Interactive dashboard with what-if analysis and team exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.mlb_data_loader import load_mlb_data
from src.data_processing.war_feature_engineering import WARFeatureEngineer


# Page configuration
st.set_page_config(
    page_title="MLB Win Rate Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    """Load and cache all MLB data."""
    batters, pitchers, standings = load_mlb_data(data_dir="data/raw", min_pa=100, min_ip=20.0)

    # Create features
    engineer = WARFeatureEngineer(k=1.83, runs_per_win=10.0)
    team_features = engineer.create_team_season_features(batters, pitchers)
    team_features = engineer.add_derived_features(team_features)

    return batters, pitchers, standings, team_features, engineer


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<div class="main-header">⚾ MLB Win Rate Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pythagorean Expectation Model Based on WAR</div>', unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading MLB data..."):
        batters, pitchers, standings, team_features, engineer = load_all_data()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "🔍 Team Explorer", "🎯 What-If Analysis", "📈 Model Diagnostics", "ℹ️ About"]
    )

    # Page routing
    if page == "📊 Overview":
        show_overview(team_features, batters, pitchers)
    elif page == "🔍 Team Explorer":
        show_team_explorer(team_features, batters, pitchers)
    elif page == "🎯 What-If Analysis":
        show_whatif_analysis(engineer)
    elif page == "📈 Model Diagnostics":
        show_diagnostics(team_features)
    elif page == "ℹ️ About":
        show_about()


def show_overview(team_features, batters, pitchers):
    """Show overview page with summary statistics."""
    st.header("📊 League Overview")

    # Season selector
    seasons = sorted(team_features['Season'].unique(), reverse=True)
    selected_season = st.selectbox("Select Season", seasons, key="overview_season")

    season_data = team_features[team_features['Season'] == selected_season]

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_off_war = season_data['Offensive_WAR'].mean()
        st.metric("Avg Offensive WAR", f"{avg_off_war:.1f}")

    with col2:
        avg_pit_war = season_data['Pitching_WAR'].mean()
        st.metric("Avg Pitching WAR", f"{avg_pit_war:.1f}")

    with col3:
        avg_pe = season_data['Pythagorean_Expectation'].mean()
        st.metric("Avg Win Rate", f"{avg_pe:.3f}")

    with col4:
        avg_wins = season_data['Expected_Wins'].mean()
        st.metric("Avg Expected Wins", f"{avg_wins:.1f}")

    st.divider()

    # Top teams
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Teams by Expected Wins")
        top_teams = season_data.nlargest(10, 'Expected_Wins')[
            ['Team', 'Offensive_WAR', 'Pitching_WAR', 'Expected_Wins', 'Pythagorean_Expectation']
        ].copy()
        top_teams['Pythagorean_Expectation'] = top_teams['Pythagorean_Expectation'].round(3)
        top_teams['Expected_Wins'] = top_teams['Expected_Wins'].round(1)
        st.dataframe(top_teams, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Bottom 10 Teams by Expected Wins")
        bottom_teams = season_data.nsmallest(10, 'Expected_Wins')[
            ['Team', 'Offensive_WAR', 'Pitching_WAR', 'Expected_Wins', 'Pythagorean_Expectation']
        ].copy()
        bottom_teams['Pythagorean_Expectation'] = bottom_teams['Pythagorean_Expectation'].round(3)
        bottom_teams['Expected_Wins'] = bottom_teams['Expected_Wins'].round(1)
        st.dataframe(bottom_teams, use_container_width=True, hide_index=True)

    st.divider()

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # WAR distribution
        # Add size_abs column for plotting (size must be positive)
        season_data_plot = season_data.copy()
        season_data_plot['Size_Metric'] = season_data_plot['Total_WAR'].abs() + 1  # Ensure positive values

        fig = px.scatter(season_data_plot, x='Offensive_WAR', y='Pitching_WAR',
                        hover_data=['Team', 'Total_WAR'], size='Size_Metric',
                        title=f'{selected_season} Team WAR Distribution',
                        labels={'Offensive_WAR': 'Offensive WAR', 'Pitching_WAR': 'Pitching WAR'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pythagorean expectation
        season_data_sorted = season_data.sort_values('Pythagorean_Expectation', ascending=False)
        fig = px.bar(season_data_sorted, x='Team', y='Pythagorean_Expectation',
                    title=f'{selected_season} Expected Win Rates by Team',
                    labels={'Pythagorean_Expectation': 'Expected Win Rate'})
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="0.500")
        fig.update_layout(height=500, xaxis={'tickangle': -45})
        st.plotly_chart(fig, use_container_width=True)


def show_team_explorer(team_features, batters, pitchers):
    """Show detailed team analysis page."""
    st.header("🔍 Team Explorer")

    # Team and season selection
    col1, col2 = st.columns(2)

    with col1:
        teams = sorted(team_features['Team'].unique())
        selected_team = st.selectbox("Select Team", teams, key="explorer_team")

    with col2:
        seasons = sorted(team_features['Season'].unique(), reverse=True)
        selected_season = st.selectbox("Select Season", seasons, key="explorer_season")

    # Get team data
    team_data = team_features[
        (team_features['Team'] == selected_team) &
        (team_features['Season'] == selected_season)
    ]

    if team_data.empty:
        st.warning(f"No data available for {selected_team} in {selected_season}")
        return

    team_row = team_data.iloc[0]

    # Display key metrics
    st.subheader(f"{selected_team} - {selected_season} Season")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Offensive WAR", f"{team_row['Offensive_WAR']:.1f}")

    with col2:
        st.metric("Pitching WAR", f"{team_row['Pitching_WAR']:.1f}")

    with col3:
        st.metric("Total WAR", f"{team_row['Total_WAR']:.1f}")

    with col4:
        st.metric("Expected Win Rate", f"{team_row['Pythagorean_Expectation']:.3f}")

    with col5:
        st.metric("Expected Wins", f"{team_row['Expected_Wins']:.1f}")

    st.divider()

    # Detailed stats
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Offensive Stats")
        st.metric("Runs Scored (Estimated)", f"{team_row['RS']:.0f}")
        st.metric("RS per Game", f"{team_row['RS_per_game']:.2f}")
        st.metric("Offensive Strength", f"{team_row['Offensive_Strength']:.2%}")

    with col2:
        st.subheader("Pitching Stats")
        st.metric("Runs Allowed (Estimated)", f"{team_row['RA']:.0f}")
        st.metric("RA per Game", f"{team_row['RA_per_game']:.2f}")
        st.metric("Pitching Strength", f"{team_row['Pitching_Strength']:.2%}")

    st.divider()

    # Player breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Batters")
        team_batters = batters[
            (batters['Team'] == selected_team) &
            (batters['Season'] == selected_season)
        ].nlargest(10, 'WAR')[['Player', 'WAR', 'PA', 'OPS']]
        st.dataframe(team_batters, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Top Pitchers")
        team_pitchers = pitchers[
            (pitchers['Team'] == selected_team) &
            (pitchers['Season'] == selected_season)
        ].nlargest(10, 'WAR')[['Player', 'WAR', 'IP', 'ERA']]
        st.dataframe(team_pitchers, use_container_width=True, hide_index=True)

    # Historical performance
    st.divider()
    st.subheader("Historical Performance")

    team_history = team_features[team_features['Team'] == selected_team].sort_values('Season')

    if len(team_history) > 1:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Expected Win Rate Over Time', 'WAR Components Over Time'),
            vertical_spacing=0.15
        )

        # Win rate trend
        fig.add_trace(
            go.Scatter(x=team_history['Season'], y=team_history['Pythagorean_Expectation'],
                      mode='lines+markers', name='Expected Win Rate'),
            row=1, col=1
        )

        # WAR components
        fig.add_trace(
            go.Scatter(x=team_history['Season'], y=team_history['Offensive_WAR'],
                      mode='lines+markers', name='Offensive WAR'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=team_history['Season'], y=team_history['Pitching_WAR'],
                      mode='lines+markers', name='Pitching WAR'),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Season", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate", row=1, col=1)
        fig.update_yaxes(title_text="WAR", row=2, col=1)

        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Historical data not available (only one season in dataset)")


def show_whatif_analysis(engineer):
    """Show what-if analysis page with interactive sliders."""
    st.header("🎯 What-If Analysis")

    st.markdown("""
    Adjust the sliders below to see how changes in Offensive and Pitching WAR
    affect a team's expected run production, run prevention, and win rate.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Adjust Team Performance")
        offensive_war = st.slider(
            "Offensive WAR",
            min_value=0.0,
            max_value=60.0,
            value=30.0,
            step=0.5,
            help="Total offensive WAR for the team"
        )

        pitching_war = st.slider(
            "Pitching WAR",
            min_value=0.0,
            max_value=40.0,
            value=20.0,
            step=0.5,
            help="Total pitching WAR for the team"
        )

    # Calculate metrics
    rs = engineer._calculate_runs_scored(pd.Series([offensive_war])).iloc[0]
    ra = engineer._calculate_runs_allowed(pd.Series([pitching_war])).iloc[0]
    pe = engineer._calculate_pythagorean_expectation(pd.Series([rs]), pd.Series([ra])).iloc[0]
    expected_wins = pe * 162
    run_diff = rs - ra

    with col2:
        st.subheader("Predicted Results")

        st.metric("Runs Scored (RS)", f"{rs:.0f}", f"{rs/162:.2f} per game")
        st.metric("Runs Allowed (RA)", f"{ra:.0f}", f"{ra/162:.2f} per game")
        st.metric("Run Differential", f"{run_diff:+.0f}")
        st.metric("Pythagorean Expectation", f"{pe:.3f}")
        st.metric("Expected Wins (162 games)", f"{expected_wins:.1f}")
        st.metric("Expected Record", f"{expected_wins:.0f}-{162-expected_wins:.0f}")

    st.divider()

    # Comparison with league average
    st.subheader("Comparison with League Average")

    avg_offensive_war = 30.0
    avg_pitching_war = 20.0

    comparison_data = pd.DataFrame({
        'Category': ['Your Team', 'League Average'],
        'Offensive WAR': [offensive_war, avg_offensive_war],
        'Pitching WAR': [pitching_war, avg_pitching_war],
        'Total WAR': [offensive_war + pitching_war, avg_offensive_war + avg_pitching_war],
        'Expected Wins': [expected_wins, 81.0]
    })

    fig = px.bar(comparison_data, x='Category',
                y=['Offensive WAR', 'Pitching WAR'],
                title='WAR Comparison',
                barmode='group')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Sensitivity analysis
    st.divider()
    st.subheader("Sensitivity Analysis")

    st.markdown("See how changes in WAR components affect expected wins:")

    # Generate sensitivity data
    war_range = np.arange(0, 50, 2)
    offensive_wins = []
    pitching_wins = []

    for war in war_range:
        # Varying offensive WAR
        rs_off = engineer._calculate_runs_scored(pd.Series([war])).iloc[0]
        ra_off = engineer._calculate_runs_allowed(pd.Series([pitching_war])).iloc[0]
        pe_off = engineer._calculate_pythagorean_expectation(pd.Series([rs_off]), pd.Series([ra_off])).iloc[0]
        offensive_wins.append(pe_off * 162)

        # Varying pitching WAR
        rs_pit = engineer._calculate_runs_scored(pd.Series([offensive_war])).iloc[0]
        ra_pit = engineer._calculate_runs_allowed(pd.Series([war])).iloc[0]
        pe_pit = engineer._calculate_pythagorean_expectation(pd.Series([rs_pit]), pd.Series([ra_pit])).iloc[0]
        pitching_wins.append(pe_pit * 162)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=war_range, y=offensive_wins, mode='lines', name='Offensive WAR Impact'))
    fig.add_trace(go.Scatter(x=war_range, y=pitching_wins, mode='lines', name='Pitching WAR Impact'))
    fig.update_layout(
        title='Impact of WAR on Expected Wins',
        xaxis_title='WAR Value',
        yaxis_title='Expected Wins',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def show_diagnostics(team_features):
    """Show model diagnostics and correlations."""
    st.header("📈 Model Diagnostics")

    st.subheader("Feature Correlations")

    # Correlation matrix
    numeric_cols = ['Offensive_WAR', 'Pitching_WAR', 'Total_WAR', 'RS', 'RA',
                   'Run_Differential', 'Pythagorean_Expectation', 'Expected_Wins']
    corr_matrix = team_features[numeric_cols].corr()

    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=numeric_cols,
                    y=numeric_cols,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title='Feature Correlation Heatmap')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Scatter matrix
    st.subheader("Relationship Analysis")

    selected_x = st.selectbox("X-axis", numeric_cols, index=0)
    selected_y = st.selectbox("Y-axis", numeric_cols, index=6)

    # Try to add trendline if statsmodels is available
    try:
        fig = px.scatter(team_features, x=selected_x, y=selected_y,
                        hover_data=['Team', 'Season'],
                        color='Season',
                        title=f'{selected_y} vs {selected_x}',
                        trendline='ols')
    except ImportError:
        # Fallback without trendline if statsmodels not available
        fig = px.scatter(team_features, x=selected_x, y=selected_y,
                        hover_data=['Team', 'Season'],
                        color='Season',
                        title=f'{selected_y} vs {selected_x}')
        st.info("📊 Tip: Install 'statsmodels' to see trendlines: pip install statsmodels")

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def show_about():
    """Show about page with methodology explanation."""
    st.header("ℹ️ About This Application")

    st.markdown("""
    ## MLB Win Rate Prediction Model

    This application predicts team win rates using **Pythagorean expectation** approximated from
    **Wins Above Replacement (WAR)** statistics.

    ### Methodology

    #### 1. Data Collection
    - Player-level WAR statistics for batters and pitchers (2023-2025 seasons)
    - Aggregated by team and season

    #### 2. Feature Engineering

    **Runs Scored (RS) Approximation:**
    ```
    RS = (replacement level runs for 9 positions) + (Offensive WAR × runs per win)
    ```

    **Runs Allowed (RA) Approximation:**
    ```
    RA = (league average RA) - (Pitching WAR × runs per win)
    ```

    **Pythagorean Expectation:**
    ```
    PE = RS^k / (RS^k + RA^k)
    ```
    where k ≈ 1.83 (optimized exponent for baseball)

    #### 3. Win Rate Prediction
    ```
    Expected Win Rate = Pythagorean Expectation
    Expected Wins = PE × 162 games
    ```

    ### Model Parameters
    - **k (Pythagorean exponent):** 1.83
    - **Runs per Win:** 10.0
    - **Replacement Level:** 4.1 runs per 600 PA
    - **Games per Season:** 162

    ### Key Insights
    - Offensive and Pitching WAR are strong predictors of team success
    - Pythagorean expectation provides a talent-based estimate independent of luck
    - Run differential is highly correlated with win rate

    ### Data Sources
    - MLB Batter Statistics (2023-2025)
    - MLB Pitcher Statistics (2023-2025)
    - Source: Baseball Reference

    ### Limitations
    - Model approximates RS/RA from WAR (not direct observation)
    - Does not account for:
      - Injuries and roster changes mid-season
      - Managerial decisions and strategy
      - Home-field advantage
      - Clutch performance in close games
    - Best used for season-level projections, not individual games

    ### Project Team
    - Cheng-Lun Lee
    - Wei-Chieh Tseng
    - Yun Chen
    - Karina Shih

    ### Course
    CSE482 - Sports Analytics

    ---

    **Built with:** Python, Streamlit, Plotly, scikit-learn, pandas
    """)


if __name__ == "__main__":
    main()
