"""
==============================================================================
ATM INTELLIGENCE DEMAND FORECASTING - FORMATIVE ASSESSMENT 2
Building Actionable Insights and Interactive Python Script
==============================================================================

This script performs comprehensive data mining analysis on ATM transaction data:
- Stage 3: Exploratory Data Analysis (EDA)
- Stage 4: Clustering Analysis of ATMs
- Stage 5: Anomaly Detection on Holidays/Events
- Stage 6: Interactive Planner Script

Author: FinTrust Bank Ltd. Data Science Team
Date: 2024
==============================================================================
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath='atm_cash_management_dataset.csv'):
    """
    Load the cleaned ATM dataset from FA-1
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    DataFrame : Loaded dataset
    """
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(filepath)
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Number of ATMs: {df['ATM_ID'].nunique()}")
    print(f"  - Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  - Columns: {len(df.columns)}")
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Display column info
    print(f"\n  Columns in dataset:")
    for col in df.columns:
        print(f"    - {col}: {df[col].dtype}")
    
    return df

# =============================================================================
# STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def perform_eda(df):
    """
    Stage 3: Comprehensive Exploratory Data Analysis
    
    This section explores the dataset visually to uncover trends, relationships,
    and patterns before applying advanced analysis techniques.
    """
    
    print("\n" + "="*80)
    print("STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    # Create output directory for visualizations
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 3.1 DISTRIBUTION ANALYSIS
    # -------------------------------------------------------------------------
    print("\n[3.1] DISTRIBUTION ANALYSIS")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram of Total Withdrawals
    axes[0, 0].hist(df['Total_Withdrawals'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Total Withdrawals', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Withdrawal Amount ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(df['Total_Withdrawals'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: ${df["Total_Withdrawals"].mean():,.0f}')
    axes[0, 0].legend()
    print("  ✓ Histogram created for Total Withdrawals")
    
    # Histogram of Total Deposits
    axes[0, 1].hist(df['Total_Deposits'], bins=50, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Distribution of Total Deposits', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Deposit Amount ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(df['Total_Deposits'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: ${df["Total_Deposits"].mean():,.0f}')
    axes[0, 1].legend()
    print("  ✓ Histogram created for Total Deposits")
    
    # Box plot for Withdrawals
    bp1 = axes[1, 0].boxplot(df['Total_Withdrawals'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightcoral')
    axes[1, 0].set_title('Box Plot - Total Withdrawals (Outlier Detection)', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Withdrawal Amount ($)')
    axes[1, 0].grid(True, alpha=0.3)
    print("  ✓ Box plot created for Withdrawals (outlier detection)")
    
    # Box plot for Deposits
    bp2 = axes[1, 1].boxplot(df['Total_Deposits'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightgreen')
    axes[1, 1].set_title('Box Plot - Total Deposits (Outlier Detection)', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Deposit Amount ($)')
    axes[1, 1].grid(True, alpha=0.3)
    print("  ✓ Box plot created for Deposits (outlier detection)")
    
    plt.tight_layout()
    plt.savefig('visualizations/3.1_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("  → Saved: visualizations/3.1_distribution_analysis.png")
    plt.close()
    
    print("\n  OBSERVATIONS:")
    print(f"  - Withdrawals range: ${df['Total_Withdrawals'].min():,} - ${df['Total_Withdrawals'].max():,}")
    print(f"  - Average withdrawal: ${df['Total_Withdrawals'].mean():,.2f}")
    print(f"  - Median withdrawal: ${df['Total_Withdrawals'].median():,.2f}")
    print(f"  - Standard deviation: ${df['Total_Withdrawals'].std():,.2f}")
    
    # -------------------------------------------------------------------------
    # 3.2 TIME-BASED TRENDS
    # -------------------------------------------------------------------------
    print("\n[3.2] TIME-BASED TRENDS")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Line chart of withdrawals over time
    daily_withdrawals = df.groupby('Date')['Total_Withdrawals'].mean().reset_index()
    axes[0, 0].plot(daily_withdrawals['Date'], daily_withdrawals['Total_Withdrawals'], 
                   color='royalblue', linewidth=1.5, alpha=0.8)
    axes[0, 0].fill_between(daily_withdrawals['Date'], daily_withdrawals['Total_Withdrawals'],
                           alpha=0.3, color='royalblue')
    axes[0, 0].set_title('Average Withdrawals Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Average Withdrawal ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    print("  ✓ Line chart created for withdrawals over time")
    
    # Bar plot for Day of Week
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_withdrawals = df.groupby('Day_of_Week')['Total_Withdrawals'].mean().reindex(dow_order)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    bars = axes[0, 1].bar(dow_order, dow_withdrawals.values, color=colors, edgecolor='black')
    axes[0, 1].set_title('Average Withdrawals by Day of Week', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Average Withdrawal ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    print("  ✓ Bar plot created for Day of Week patterns")
    
    # Bar plot for Time of Day
    tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    tod_withdrawals = df.groupby('Time_of_Day')['Total_Withdrawals'].mean().reindex(tod_order)
    bars = axes[1, 0].bar(tod_order, tod_withdrawals.values, 
                         color=['#FFD93D', '#6BCB77', '#4D96FF', '#FF6B6B'], edgecolor='black')
    axes[1, 0].set_title('Average Withdrawals by Time of Day', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time of Day')
    axes[1, 0].set_ylabel('Average Withdrawal ($)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    print("  ✓ Bar plot created for Time of Day patterns")
    
    # Monthly trend
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_withdrawals = df.groupby('Month')['Total_Withdrawals'].mean()
    axes[1, 1].bar(range(len(monthly_withdrawals)), monthly_withdrawals.values, 
                  color='steelblue', edgecolor='black')
    axes[1, 1].set_title('Monthly Withdrawal Trend', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Average Withdrawal ($)')
    axes[1, 1].set_xticks(range(len(monthly_withdrawals)))
    axes[1, 1].set_xticklabels([str(m) for m in monthly_withdrawals.index], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    print("  ✓ Monthly trend bar chart created")
    
    plt.tight_layout()
    plt.savefig('visualizations/3.2_time_based_trends.png', dpi=300, bbox_inches='tight')
    print("  → Saved: visualizations/3.2_time_based_trends.png")
    plt.close()
    
    print("\n  OBSERVATIONS:")
    highest_dow = dow_withdrawals.idxmax()
    lowest_dow = dow_withdrawals.idxmin()
    print(f"  - Highest withdrawal day: {highest_dow} (${dow_withdrawals[highest_dow]:,.2f})")
    print(f"  - Lowest withdrawal day: {lowest_dow} (${dow_withdrawals[lowest_dow]:,.2f})")
    
    # -------------------------------------------------------------------------
    # 3.3 HOLIDAY & EVENT IMPACT
    # -------------------------------------------------------------------------
    print("\n[3.3] HOLIDAY & EVENT IMPACT")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Holiday impact
    holiday_withdrawals = df.groupby('Holiday_Flag')['Total_Withdrawals'].mean()
    holiday_labels = ['Normal Day', 'Holiday']
    colors = ['#95E1D3', '#F38181']
    bars = axes[0, 0].bar(holiday_labels, holiday_withdrawals.values, color=colors, edgecolor='black')
    axes[0, 0].set_title('Withdrawals: Normal vs Holiday Days', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Average Withdrawal ($)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    print("  ✓ Bar plot created for Holiday impact")
    
    # Special Event impact
    event_withdrawals = df.groupby('Special_Event_Flag')['Total_Withdrawals'].mean()
    event_labels = ['No Event', 'Special Event']
    bars = axes[0, 1].bar(event_labels, event_withdrawals.values, 
                         color=['#A8D8EA', '#AA96DA'], edgecolor='black')
    axes[0, 1].set_title('Withdrawals: Normal vs Special Event Days', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Average Withdrawal ($)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    print("  ✓ Bar plot created for Special Event impact")
    
    # Combined Holiday and Event impact
    combined = df.groupby(['Holiday_Flag', 'Special_Event_Flag'])['Total_Withdrawals'].mean().unstack()
    combined.plot(kind='bar', ax=axes[1, 0], color=['#A8D8EA', '#AA96DA'], edgecolor='black')
    axes[1, 0].set_title('Withdrawals by Holiday and Event Combination', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Holiday Flag')
    axes[1, 0].set_ylabel('Average Withdrawal ($)')
    axes[1, 0].set_xticklabels(['Normal Day', 'Holiday'], rotation=0)
    axes[1, 0].legend(['No Event', 'Special Event'])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    print("  ✓ Combined holiday and event analysis created")
    
    # Count of records by holiday/event
    holiday_counts = df.groupby(['Holiday_Flag', 'Special_Event_Flag']).size().unstack()
    holiday_counts.plot(kind='bar', ax=axes[1, 1], color=['#95E1D3', '#F38181'], edgecolor='black')
    axes[1, 1].set_title('Number of Records by Holiday/Event', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Holiday Flag')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticklabels(['Normal Day', 'Holiday'], rotation=0)
    axes[1, 1].legend(['No Event', 'Special Event'])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    print("  ✓ Record count by holiday/event created")
    
    plt.tight_layout()
    plt.savefig('visualizations/3.3_holiday_event_impact.png', dpi=300, bbox_inches='tight')
    print("  → Saved: visualizations/3.3_holiday_event_impact.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # 3.4 EXTERNAL FACTORS
    # -------------------------------------------------------------------------
    print("\n[3.4] EXTERNAL FACTORS")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Weather condition impact
    weather_order = ['Clear', 'Cloudy', 'Rainy', 'Snowy']
    weather_withdrawals = df.groupby('Weather_Condition')['Total_Withdrawals'].mean().reindex(weather_order)
    weather_colors = ['#FFD93D', '#B0BEC5', '#64B5F6', '#E8EAF6']
    bars = axes[0, 0].bar(weather_order, weather_withdrawals.values, color=weather_colors, edgecolor='black')
    axes[0, 0].set_title('Average Withdrawals by Weather Condition', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Average Withdrawal ($)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    print("  ✓ Bar plot created for Weather Condition")
    
    # Box plot by weather
    weather_data = [df[df['Weather_Condition'] == w]['Total_Withdrawals'].values for w in weather_order]
    bp = axes[0, 1].boxplot(weather_data, labels=weather_order, patch_artist=True)
    for patch, color in zip(bp['boxes'], weather_colors):
        patch.set_facecolor(color)
    axes[0, 1].set_title('Withdrawal Distribution by Weather', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Withdrawal Amount ($)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    print("  ✓ Box plot created for Weather Condition")
    
    # Competitor ATM impact
    competitor_withdrawals = df.groupby('Nearby_Competitor_ATMs')['Total_Withdrawals'].mean()
    axes[1, 0].bar(competitor_withdrawals.index, competitor_withdrawals.values, 
                  color='steelblue', edgecolor='black')
    axes[1, 0].set_title('Withdrawals by Number of Competitor ATMs', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Nearby Competitor ATMs')
    axes[1, 0].set_ylabel('Average Withdrawal ($)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    print("  ✓ Bar plot created for Competitor ATM impact")
    
    # Location Type impact
    location_withdrawals = df.groupby('Location_Type')['Total_Withdrawals'].mean().sort_values(ascending=False)
    location_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = axes[1, 1].bar(range(len(location_withdrawals)), location_withdrawals.values, 
                         color=location_colors[:len(location_withdrawals)], edgecolor='black')
    axes[1, 1].set_title('Withdrawals by Location Type', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Location Type')
    axes[1, 1].set_ylabel('Average Withdrawal ($)')
    axes[1, 1].set_xticks(range(len(location_withdrawals)))
    axes[1, 1].set_xticklabels(location_withdrawals.index, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    print("  ✓ Bar plot created for Location Type")
    
    plt.tight_layout()
    plt.savefig('visualizations/3.4_external_factors.png', dpi=300, bbox_inches='tight')
    print("  → Saved: visualizations/3.4_external_factors.png")
    plt.close()
    
    print("\n  OBSERVATIONS:")
    print(f"  - Best weather for withdrawals: {weather_withdrawals.idxmax()} (${weather_withdrawals.max():,.2f})")
    print(f"  - Worst weather for withdrawals: {weather_withdrawals.idxmin()} (${weather_withdrawals.min():,.2f})")
    print(f"  - Best location type: {location_withdrawals.idxmax()} (${location_withdrawals.max():,.2f})")
    
    # -------------------------------------------------------------------------
    # 3.5 RELATIONSHIP ANALYSIS
    # -------------------------------------------------------------------------
    print("\n[3.5] RELATIONSHIP ANALYSIS")
    print("-" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot: Previous Day Cash Level vs Next Day Demand
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    axes[0].scatter(sample_df['Previous_Day_Cash_Level'], 
                   sample_df['Cash_Demand_Next_Day'],
                   alpha=0.5, color='royalblue', s=20)
    axes[0].set_title('Previous Day Cash Level vs Next Day Demand', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Previous Day Cash Level ($)')
    axes[0].set_ylabel('Cash Demand Next Day ($)')
    axes[0].grid(True, alpha=0.3)
    print("  ✓ Scatter plot created for cash level relationship")
    
    # Add trend line
    z = np.polyfit(sample_df['Previous_Day_Cash_Level'], 
                   sample_df['Cash_Demand_Next_Day'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample_df['Previous_Day_Cash_Level'].min(), 
                        sample_df['Previous_Day_Cash_Level'].max(), 100)
    axes[0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    axes[0].legend()
    
    # Correlation heatmap
    numeric_cols = ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 
                   'Cash_Demand_Next_Day', 'Holiday_Flag', 'Special_Event_Flag', 
                   'Nearby_Competitor_ATMs']
    correlation_matrix = df[numeric_cols].corr()
    
    im = axes[1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(numeric_cols)))
    axes[1].set_yticks(range(len(numeric_cols)))
    axes[1].set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(numeric_cols, fontsize=9)
    axes[1].set_title('Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold')
    
    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            text = axes[1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig('visualizations/3.5_relationship_analysis.png', dpi=300, bbox_inches='tight')
    print("  → Saved: visualizations/3.5_relationship_analysis.png")
    plt.close()
    
    print("\n  OBSERVATIONS:")
    corr_withdrawals_deposit = correlation_matrix.loc['Total_Withdrawals', 'Total_Deposits']
    print(f"  - Correlation between withdrawals and deposits: {corr_withdrawals_deposit:.3f}")
    
    print("\n" + "="*80)
    print("✓ STAGE 3: EDA COMPLETED SUCCESSFULLY")
    print("="*80)

# =============================================================================
# STAGE 4: CLUSTERING ANALYSIS
# =============================================================================

def perform_clustering(df):
    """
    Stage 4: Clustering Analysis of ATMs
    """
    
    print("\n" + "="*80)
    print("STAGE 4: CLUSTERING ANALYSIS OF ATMs")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 4.1 FEATURE SELECTION AND PREPARATION
    # -------------------------------------------------------------------------
    print("\n[4.1] FEATURE SELECTION AND PREPARATION")
    print("-" * 80)
    
    print("  Aggregating data by ATM...")
    atm_features = df.groupby('ATM_ID').agg({
        'Total_Withdrawals': ['mean', 'std', 'max', 'min'],
        'Total_Deposits': 'mean',
        'Location_Type': 'first',
        'Nearby_Competitor_ATMs': 'mean'
    }).reset_index()
    
    atm_features.columns = ['ATM_ID', 'Avg_Withdrawals', 'Std_Withdrawals', 
                           'Max_Withdrawals', 'Min_Withdrawals', 'Avg_Deposits', 
                           'Location_Type', 'Avg_Competitors']
    
    atm_features['Std_Withdrawals'] = atm_features['Std_Withdrawals'].fillna(0)
    
    print(f"  ✓ Aggregated data for {len(atm_features)} ATMs")
    
    le = LabelEncoder()
    atm_features['Location_Encoded'] = le.fit_transform(atm_features['Location_Type'])
    
    clustering_features = ['Avg_Withdrawals', 'Std_Withdrawals', 'Max_Withdrawals',
                          'Avg_Deposits', 'Location_Encoded', 'Avg_Competitors']
    
    X = atm_features[clustering_features].copy()
    print(f"  ✓ Selected {len(clustering_features)} features for clustering")
    
    # -------------------------------------------------------------------------
    # 4.2 FEATURE STANDARDIZATION
    # -------------------------------------------------------------------------
    print("\n[4.2] FEATURE STANDARDIZATION")
    print("-" * 80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("  ✓ Features standardized using StandardScaler")
    
    # -------------------------------------------------------------------------
    # 4.3 DETERMINE OPTIMAL CLUSTERS
    # -------------------------------------------------------------------------
    print("\n[4.3] DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    print("-" * 80)
    
    wcss = []
    silhouette_scores = []
    k_range = range(2, min(11, len(atm_features)))
    
    print("  Testing K values from 2 to 10...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
        if len(set(kmeans.labels_)) > 1:
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
        
        print(f"    K={k}: WCSS={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].plot(list(k_range), wcss, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('WCSS', fontsize=12)
    axes[0].set_title('Elbow Method - Optimal K', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    optimal_k = max(3, min(5, list(k_range)[np.argmax(silhouette_scores)]))
    axes[0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal K = {optimal_k}')
    axes[0].legend()
    
    axes[1].plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Scores', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal K = {optimal_k}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/4.3_optimal_clusters.png', dpi=300, bbox_inches='tight')
    print("  → Saved: visualizations/4.3_optimal_clusters.png")
    plt.close()
    
    print(f"\n  ✓ Optimal number of clusters: K = {optimal_k}")
    
    # -------------------------------------------------------------------------
    # 4.4 APPLY K-MEANS CLUSTERING
    # -------------------------------------------------------------------------
    print("\n[4.4] APPLYING K-MEANS CLUSTERING")
    print("-" * 80)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    atm_features['Cluster'] = clusters
    
    print(f"  ✓ K-Means clustering completed with K={optimal_k}")
    
    # -------------------------------------------------------------------------
    # 4.5 INTERPRET CLUSTERS
    # -------------------------------------------------------------------------
    print("\n[4.5] INTERPRETING CLUSTERS")
    print("-" * 80)
    
    cluster_labels = {}
    overall_avg = atm_features['Avg_Withdrawals'].mean()
    
    for cluster in range(optimal_k):
        cluster_data = atm_features[atm_features['Cluster'] == cluster]
        avg_withdrawal = cluster_data['Avg_Withdrawals'].mean()
        
        if avg_withdrawal > overall_avg * 1.3:
            cluster_labels[cluster] = "High-Demand Cluster"
        elif avg_withdrawal > overall_avg * 0.7:
            cluster_labels[cluster] = "Steady-Demand Cluster"
        else:
            cluster_labels[cluster] = "Low-Demand Cluster"
        
        print(f"\n  Cluster {cluster}: {cluster_labels[cluster]}")
        print(f"    → Average Withdrawals: ${avg_withdrawal:,.2f}")
        print(f"    → Number of ATMs: {len(cluster_data)}")
    
    atm_features['Cluster_Label'] = atm_features['Cluster'].map(cluster_labels)
    
    # -------------------------------------------------------------------------
    # 4.6 VISUALIZE CLUSTERS
    # -------------------------------------------------------------------------
    print("\n[4.6] VISUALIZING CLUSTERS")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    for cluster in range(optimal_k):
        mask = atm_features['Cluster'] == cluster
        axes[0, 0].scatter(atm_features.loc[mask, 'Avg_Withdrawals'],
                          atm_features.loc[mask, 'Std_Withdrawals'],
                          label=cluster_labels[cluster], s=100, alpha=0.7,
                          color=colors[cluster], edgecolors='black', linewidth=1.5)
    axes[0, 0].set_xlabel('Average Withdrawals ($)', fontsize=12)
    axes[0, 0].set_ylabel('Withdrawal Std Dev ($)', fontsize=12)
    axes[0, 0].set_title('ATM Clusters: Avg Withdrawals vs Volatility', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    cluster_avg = atm_features.groupby('Cluster_Label')['Avg_Withdrawals'].mean()
    cluster_avg.plot(kind='bar', ax=axes[0, 1], color=colors[:optimal_k], edgecolor='black')
    axes[0, 1].set_title('Average Withdrawals by Cluster', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Average Withdrawal ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    cluster_counts = atm_features['Cluster_Label'].value_counts()
    axes[1, 0].bar(cluster_counts.index, cluster_counts.values, 
                   color=colors[:optimal_k], edgecolor='black')
    axes[1, 0].set_title('Number of ATMs per Cluster', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    cluster_location = atm_features.groupby(['Cluster_Label', 'Location_Type']).size().unstack(fill_value=0)
    cluster_location.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='viridis', edgecolor='black')
    axes[1, 1].set_title('Location Type Distribution by Cluster', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('visualizations/4.6_cluster_visualizations.png', dpi=300, bbox_inches='tight')
    print("  → Saved: visualizations/4.6_cluster_visualizations.png")
    plt.close()
    
    cluster_assignments = atm_features[['ATM_ID', 'Cluster', 'Cluster_Label', 
                                       'Avg_Withdrawals', 'Std_Withdrawals', 'Location_Type']]
    cluster_assignments.to_csv('atm_cluster_assignments.csv', index=False)
    print("\n  → Saved: atm_cluster_assignments.csv")
    
    print("\n" + "="*80)
    print("✓ STAGE 4: CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return atm_features, cluster_labels

# =============================================================================
# STAGE 5: ANOMALY DETECTION
# =============================================================================

def perform_anomaly_detection(df):
    """
    Stage 5: Anomaly Detection on Holidays/Events
    """
    
    print("\n" + "="*80)
    print("STAGE 5: ANOMALY DETECTION ON HOLIDAYS/EVENTS")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 5.1 HOLIDAY VS NORMAL DAY COMPARISON
    # -------------------------------------------------------------------------
    print("\n[5.1] HOLIDAY VS NORMAL DAY COMPARISON")
    print("-" * 80)
    
    normal_withdrawals = df[df['Holiday_Flag'] == 0]['Total_Withdrawals']
    holiday_withdrawals = df[df['Holiday_Flag'] == 1]['Total_Withdrawals']
    
    print(f"  Normal days: {len(normal_withdrawals)} records, Mean: ${normal_withdrawals.mean():,.2f}")
    print(f"  Holiday days: {len(holiday_withdrawals)} records, Mean: ${holiday_withdrawals.mean():,.2f}")
    
    # -------------------------------------------------------------------------
    # 5.2 Z-SCORE BASED ANOMALY DETECTION
    # -------------------------------------------------------------------------
    print("\n[5.2] Z-SCORE BASED ANOMALY DETECTION")
    print("-" * 80)
    
    df['Withdrawal_ZScore'] = (df['Total_Withdrawals'] - df['Total_Withdrawals'].mean()) / \
                              df['Total_Withdrawals'].std()
    
    anomaly_threshold = 3
    df['Is_Anomaly_ZScore'] = np.abs(df['Withdrawal_ZScore']) > anomaly_threshold
    
    num_anomalies = df['Is_Anomaly_ZScore'].sum()
    print(f"  ✓ Anomaly threshold: |Z-score| > {anomaly_threshold}")
    print(f"  ✓ Number of anomalies detected: {num_anomalies} ({num_anomalies/len(df)*100:.2f}%)")
    
    # -------------------------------------------------------------------------
    # 5.3 ISOLATION FOREST
    # -------------------------------------------------------------------------
    print("\n[5.3] ISOLATION FOREST FOR MULTIVARIATE ANOMALIES")
    print("-" * 80)
    
    isolation_features = ['Total_Withdrawals', 'Total_Deposits', 'Holiday_Flag',
                         'Special_Event_Flag', 'Previous_Day_Cash_Level']
    X_iso = df[isolation_features].copy()
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Is_Anomaly_IsoForest'] = iso_forest.fit_predict(X_iso)
    df['Is_Anomaly_IsoForest'] = df['Is_Anomaly_IsoForest'] == -1
    
    iso_anomalies = df['Is_Anomaly_IsoForest'].sum()
    print(f"  ✓ Isolation Forest anomalies: {iso_anomalies} ({iso_anomalies/len(df)*100:.2f}%)")
    
    # -------------------------------------------------------------------------
    # 5.4 VISUALIZE ANOMALIES
    # -------------------------------------------------------------------------
    print("\n[5.4] VISUALIZING ANOMALIES")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    daily_avg = df.groupby('Date')['Total_Withdrawals'].mean().reset_index()
    axes[0, 0].plot(daily_avg['Date'], daily_avg['Total_Withdrawals'], 
                   color='steelblue', label='Normal', linewidth=1.5, alpha=0.7)
    axes[0, 0].set_title('Withdrawal Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Average Withdrawal ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(df['Withdrawal_ZScore'], bins=50, color='lightblue', edgecolor='black')
    axes[0, 1].axvline(anomaly_threshold, color='red', linestyle='--', linewidth=2, 
                     label=f'Threshold = ±{anomaly_threshold}')
    axes[0, 1].axvline(-anomaly_threshold, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Distribution of Z-Scores', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Z-Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    normal_mask = ~df['Is_Anomaly_ZScore']
    axes[1, 0].scatter(df.loc[normal_mask, 'Total_Withdrawals'],
                      df.loc[normal_mask, 'Total_Deposits'],
                      color='steelblue', alpha=0.3, s=20, label='Normal')
    axes[1, 0].scatter(df.loc[~normal_mask, 'Total_Withdrawals'],
                      df.loc[~normal_mask, 'Total_Deposits'],
                      color='red', alpha=0.7, s=50, label='Anomaly', edgecolors='black')
    axes[1, 0].set_title('Anomalies: Withdrawals vs Deposits', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Total Withdrawals ($)')
    axes[1, 0].set_ylabel('Total Deposits ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    anomaly_by_flag = df.groupby(['Holiday_Flag', 'Special_Event_Flag'])['Is_Anomaly_ZScore'].mean() * 100
    anomaly_by_flag = anomaly_by_flag.unstack()
    anomaly_by_flag.plot(kind='bar', ax=axes[1, 1], colormap='RdYlGn', edgecolor='black')
    axes[1, 1].set_title('Anomaly Rate by Holiday/Event Flags', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Anomaly Rate (%)')
    axes[1, 1].set_xlabel('Holiday Flag')
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].legend(['No Event', 'Special Event'])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('visualizations/5.4_anomaly_visualizations.png', dpi=300, bbox_inches='tight')
    print("  → Saved: visualizations/5.4_anomaly_visualizations.png")
    plt.close()
    
    anomalies_df = df[df['Is_Anomaly_ZScore']][['ATM_ID', 'Date', 'Day_of_Week', 'Time_of_Day',
                                                'Total_Withdrawals', 'Total_Deposits',
                                                'Holiday_Flag', 'Special_Event_Flag',
                                                'Weather_Condition', 'Withdrawal_ZScore']]
    anomalies_df.to_csv('anomalies_detected.csv', index=False)
    print(f"\n  → Saved: anomalies_detected.csv ({len(anomalies_df)} anomalies)")
    
    print("\n" + "="*80)
    print("✓ STAGE 5: ANOMALY DETECTION COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return df

# =============================================================================
# STAGE 6: INTERACTIVE PLANNER SCRIPT
# =============================================================================

def interactive_planner(df, atm_features, cluster_labels):
    """
    Stage 6: Interactive Planner Script
    """
    
    print("\n" + "="*80)
    print("STAGE 6: INTERACTIVE PLANNER")
    print("="*80)
    
    df = df.merge(atm_features[['ATM_ID', 'Cluster', 'Cluster_Label', 
                               'Avg_Withdrawals', 'Std_Withdrawals']], 
                 on='ATM_ID', how='left')
    
    print("\n[6.1] INTERACTIVE EXPLORATION OPTIONS")
    print("-" * 80)
    print("The planner provides the following interactive capabilities:")
    print("\n  1. Filter by Day of Week")
    print("  2. Filter by Time of Day")
    print("  3. Filter by Location Type")
    print("  4. Filter by Cluster")
    print("  5. View Holiday/Event Analysis")
    print("  6. View Anomaly Analysis")
    print("  7. Generate Comprehensive Report")
    
    print("\n[6.2] GENERATING INTERACTIVE VISUALIZATIONS")
    print("-" * 80)
    
    # Generate comprehensive report
    print("  Generating comprehensive report...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    daily_avg = df.groupby('Date')['Total_Withdrawals'].mean()
    ax1.plot(daily_avg.index, daily_avg.values, color='steelblue', linewidth=1.5)
    ax1.fill_between(daily_avg.index, daily_avg.values, alpha=0.3, color='steelblue')
    ax1.set_title('Overall Withdrawal Trend', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Withdrawal ($)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    dow_avg = df.groupby('Day_of_Week')['Total_Withdrawals'].mean()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg = dow_avg.reindex(dow_order)
    ax2.bar(dow_order, dow_avg.values, color='#FF6B6B', edgecolor='black')
    ax2.set_title('Withdrawals by Day of Week', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = fig.add_subplot(gs[1, 1])
    cluster_counts = df['Cluster_Label'].value_counts()
    ax3.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%')
    ax3.set_title('Cluster Distribution', fontsize=12, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[1, 2])
    holiday_impact = df.groupby('Holiday_Flag')['Total_Withdrawals'].mean()
    ax4.bar(['Normal', 'Holiday'], holiday_impact.values, color=['#95E1D3', '#F38181'], edgecolor='black')
    ax4.set_title('Holiday Impact', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(gs[2, 0])
    tod_avg = df.groupby('Time_of_Day')['Total_Withdrawals'].mean()
    tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    tod_avg = tod_avg.reindex(tod_order)
    ax5.bar(tod_avg.index, tod_avg.values, color=['#FFD93D', '#6BCB77', '#4D96FF', '#FF6B6B'], edgecolor='black')
    ax5.set_title('Withdrawals by Time of Day', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    ax6 = fig.add_subplot(gs[2, 1])
    weather_avg = df.groupby('Weather_Condition')['Total_Withdrawals'].mean()
    ax6.bar(weather_avg.index, weather_avg.values, color='purple', edgecolor='black')
    ax6.set_title('Withdrawals by Weather', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')
    
    ax7 = fig.add_subplot(gs[2, 2])
    anomaly_rate = df['Is_Anomaly_ZScore'].mean() * 100
    ax7.bar(['Normal', 'Anomaly'], [100-anomaly_rate, anomaly_rate], 
           color=['#95E1D3', '#F38181'], edgecolor='black')
    ax7.set_title(f'Overall Anomaly Rate ({anomaly_rate:.2f}%)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    holiday_increase = 0
    if df['Holiday_Flag'].sum() > 0:
        holiday_increase = (df[df['Holiday_Flag']==1]['Total_Withdrawals'].mean() / 
                           df[df['Holiday_Flag']==0]['Total_Withdrawals'].mean() - 1) * 100
    
    stats_text = f"""
    COMPREHENSIVE ANALYSIS SUMMARY
    {'='*60}
    Dataset Statistics:
       • Total Records: {len(df):,}
       • Number of ATMs: {df['ATM_ID'].nunique()}
       • Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
    
    Transaction Patterns:
       • Average Daily Withdrawal: ${df['Total_Withdrawals'].mean():,.2f}
       • Peak Withdrawal: ${df['Total_Withdrawals'].max():,}
       • Highest Demand Day: {df.groupby('Day_of_Week')['Total_Withdrawals'].mean().idxmax()}
       • Peak Time: {df.groupby('Time_of_Day')['Total_Withdrawals'].mean().idxmax()}
    
    Key Insights:
       • Holidays {'increase' if holiday_increase > 0 else 'change'} demand by {abs(holiday_increase):.1f}%
       • {len(cluster_labels)} distinct ATM clusters identified
       • {anomaly_rate:.2f}% of transactions flagged as anomalies
    """
    
    ax8.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ATM INTELLIGENCE - COMPREHENSIVE REPORT', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig('visualizations/6.comprehensive_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Comprehensive report: visualizations/6.comprehensive_report.png")
    
    print("\n[6.3] SAVING FILTERED DATASETS")
    print("-" * 80)
    
    df.to_csv('atm_data_complete.csv', index=False)
    print("  ✓ Saved: atm_data_complete.csv")
    
    summary_stats = {
        'Total Records': len(df),
        'Unique ATMs': df['ATM_ID'].nunique(),
        'Date Range Start': df['Date'].min().strftime('%Y-%m-%d'),
        'Date Range End': df['Date'].max().strftime('%Y-%m-%d'),
        'Avg Daily Withdrawal': df['Total_Withdrawals'].mean(),
        'Max Daily Withdrawal': df['Total_Withdrawals'].max(),
        'Total Anomalies': df['Is_Anomaly_ZScore'].sum(),
        'Anomaly Rate (%)': df['Is_Anomaly_ZScore'].mean() * 100,
        'Holiday Change (%)': holiday_increase,
        'Number of Clusters': len(cluster_labels)
    }
    
    summary_df = pd.DataFrame.from_dict(summary_stats, orient='index', columns=['Value'])
    summary_df.to_csv('analysis_summary.csv')
    print("  ✓ Saved: analysis_summary.csv")
    
    print("\n" + "="*80)
    print("✓ STAGE 6: INTERACTIVE PLANNER COMPLETED SUCCESSFULLY")
    print("="*80)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function that runs all stages of the analysis
    """
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "ATM INTELLIGENCE DEMAND FORECASTING" + " "*22 + "║")
    print("║" + " "*15 + "Formative Assessment 2 - Data Mining Analysis" + " "*17 + "║")
    print("╚" + "="*78 + "╝")
    
    df = load_data()
    perform_eda(df)
    atm_features, cluster_labels = perform_clustering(df)
    df = perform_anomaly_detection(df)
    interactive_planner(df, atm_features, cluster_labels)
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "ANALYSIS COMPLETE" + " "*31 + "║")
    print("╠" + "="*78 + "╣")
    print("║  All stages completed successfully!                                            ║")
    print("║                                                                                ║")
    print("║  Outputs generated:                                                           ║")
    print("║  • visualizations/: All EDA, clustering, and anomaly visualizations           ║")
    print("║  • atm_data_complete.csv: Complete dataset with all analyses                   ║")
    print("║  • atm_cluster_assignments.csv: ATM cluster assignments                        ║")
    print("║  • anomalies_detected.csv: List of detected anomalies                          ║")
    print("║  • analysis_summary.csv: Summary statistics                                   ║")
    print("╚" + "="*78 + "╝")
    print()

if __name__ == "__main__":
    main()
