"""
==============================================================================
ATM INTELLIGENCE DEMAND FORECASTING - STREAMLIT DASHBOARD
Formative Assessment 2 - Interactive Data Mining Application
==============================================================================

This Streamlit application provides an interactive interface for:
- Stage 3: Exploratory Data Analysis (EDA)
- Stage 4: Clustering Analysis of ATMs
- Stage 5: Anomaly Detection on Holidays/Events
- Stage 6: Interactive Planner

Author: FinTrust Bank Ltd. Data Science Team
==============================================================================
"""

import streamlit as st
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

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ATM Intelligence Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #E0F2FE, #BAE6FD);
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E40AF;
        padding: 10px;
        border-left: 5px solid #3B82F6;
        background-color: #EFF6FF;
        margin: 15px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px;
    }
    .insight-box {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stSidebar {
        background-color: #F8FAFC;
    }
    .mascot-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        background: white;
        padding: 10px;
        border-radius: 50%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================
@st.cache_data
def load_data():
    """Load and cache the ATM dataset"""
    df = pd.read_csv('atm_cash_management_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
def create_sidebar():
    """Create the sidebar with navigation and filters"""
    st.sidebar.title("🏦 ATM Intelligence")
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.header("📋 Navigation")
    page = st.sidebar.radio(
        "Select Analysis Stage:",
        ["🏠 Home", "📊 EDA", "🎯 Clustering", "🔍 Anomaly Detection", "📈 Interactive Planner"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Global Filters
    st.sidebar.header("🔧 Global Filters")
    
    df = load_data()
    
    # Date range filter
    min_date = df['Date'].min().to_pydatetime()
    max_date = df['Date'].max().to_pydatetime()
    
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # ATM selection
    all_atms = ['All ATMs'] + sorted(df['ATM_ID'].unique().tolist())
    selected_atm = st.sidebar.selectbox("Select ATM:", all_atms)
    
    # Location type filter
    location_types = ['All Locations'] + df['Location_Type'].unique().tolist()
    selected_location = st.sidebar.selectbox("Select Location Type:", location_types)
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    **ATM Intelligence Dashboard**
    
    This application provides comprehensive 
    data mining analysis for ATM cash 
    demand forecasting.
    
    **Features:**
    - EDA Visualizations
    - K-Means Clustering
    - Anomaly Detection
    - Interactive Filtering
    """)
    
    return page, date_range, selected_atm, selected_location

# =============================================================================
# FILTER DATA FUNCTION
# =============================================================================
def filter_data(df, date_range, selected_atm, selected_location):
    """Filter the dataset based on user selections"""
    filtered_df = df.copy()
    
    # Apply date filter
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & 
                                  (filtered_df['Date'] <= end_date)]
    
    # Apply ATM filter
    if selected_atm != 'All ATMs':
        filtered_df = filtered_df[filtered_df['ATM_ID'] == selected_atm]
    
    # Apply location filter
    if selected_location != 'All Locations':
        filtered_df = filtered_df[filtered_df['Location_Type'] == selected_location]
    
    return filtered_df

# =============================================================================
# HOME PAGE
# =============================================================================
def show_home_page(df):
    """Display the home page with overview metrics"""
    st.markdown('<h1 class="main-header">🏦 ATM Intelligence Demand Forecasting</h1>', 
                unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">📊 Dashboard Overview</h2>', 
                unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{len(df):,}",
            delta="Historical Data"
        )
    
    with col2:
        st.metric(
            label="Number of ATMs",
            value=df['ATM_ID'].nunique(),
            delta="Active Locations"
        )
    
    with col3:
        st.metric(
            label="Avg Withdrawal",
            value=f"${df['Total_Withdrawals'].mean():,.0f}",
            delta="Daily Average"
        )
    
    with col4:
        st.metric(
            label="Total Withdrawals",
            value=f"${df['Total_Withdrawals'].sum():,.0f}",
            delta="Cumulative"
        )
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown('<h3 class="sub-header">📈 Quick Statistics</h3>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Transaction Summary")
        summary_df = pd.DataFrame({
            'Metric': ['Min Withdrawal', 'Max Withdrawal', 'Median Withdrawal', 
                      'Std Deviation', 'Total Deposits'],
            'Value': [f"${df['Total_Withdrawals'].min():,.0f}",
                     f"${df['Total_Withdrawals'].max():,.0f}",
                     f"${df['Total_Withdrawals'].median():,.0f}",
                     f"${df['Total_Withdrawals'].std():,.0f}",
                     f"${df['Total_Deposits'].sum():,.0f}"]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Data Distribution")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        location_counts = df['Location_Type'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        ax.pie(location_counts.values, labels=location_counts.index, autopct='%1.1f%%',
               colors=colors[:len(location_counts)], startangle=90)
        ax.set_title('ATM Distribution by Location Type', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Timeline
    st.markdown('<h3 class="sub-header">📅 Withdrawal Timeline</h3>', 
                unsafe_allow_html=True)
    
    daily_withdrawals = df.groupby('Date')['Total_Withdrawals'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_withdrawals['Date'], daily_withdrawals['Total_Withdrawals'], 
            color='steelblue', linewidth=1.5, alpha=0.8)
    ax.fill_between(daily_withdrawals['Date'], daily_withdrawals['Total_Withdrawals'],
                   alpha=0.3, color='steelblue')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
    ax.set_title('Average Daily Withdrawals Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Key Insights
    st.markdown('<h3 class="sub-header">💡 Key Insights</h3>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4>🏆 Top Performance</h4>
            <p><b>Best Day:</b> {}</p>
            <p><b>Best Time:</b> {}</p>
            <p><b>Best Weather:</b> {}</p>
        </div>
        """.format(
            df.groupby('Day_of_Week')['Total_Withdrawals'].mean().idxmax(),
            df.groupby('Time_of_Day')['Total_Withdrawals'].mean().idxmax(),
            df.groupby('Weather_Condition')['Total_Withdrawals'].mean().idxmax()
        ), unsafe_allow_html=True)
    
    with col2:
        holiday_impact = df[df['Holiday_Flag']==1]['Total_Withdrawals'].mean()
        normal_impact = df[df['Holiday_Flag']==0]['Total_Withdrawals'].mean()
        holiday_change = ((holiday_impact - normal_impact) / normal_impact) * 100
        
        st.markdown("""
        <div class="insight-box">
            <h4>🎉 Holiday Impact</h4>
            <p><b>Normal Days:</b> ${:,.0f}</p>
            <p><b>Holidays:</b> ${:,.0f}</p>
            <p><b>Change:</b> {:.1f}%</p>
        </div>
        """.format(normal_impact, holiday_impact, holiday_change), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-box">
            <h4>📍 Location Insights</h4>
            <p><b>Top Location:</b> {}</p>
            <p><b>Total Locations:</b> {}</p>
            <p><b>Weather Types:</b> {}</p>
        </div>
        """.format(
            df.groupby('Location_Type')['Total_Withdrawals'].mean().idxmax(),
            df['Location_Type'].nunique(),
            df['Weather_Condition'].nunique()
        ), unsafe_allow_html=True)

# =============================================================================
# EDA PAGE
# =============================================================================
def show_eda_page(df):
    """Display the Exploratory Data Analysis page"""
    st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Tabs for different EDA sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distribution", "⏰ Time Trends", "🎉 Holiday & Events", 
        "🌤️ External Factors", "🔗 Relationships"
    ])
    
    # Tab 1: Distribution Analysis
    with tab1:
        st.markdown('<h3 class="sub-header">Distribution Analysis</h3>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Total Withdrawals Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['Total_Withdrawals'], bins=50, color='skyblue', edgecolor='black')
            ax.axvline(df['Total_Withdrawals'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: ${df["Total_Withdrawals"].mean():,.0f}')
            ax.set_xlabel('Withdrawal Amount ($)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Total Withdrawals', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Total Deposits Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['Total_Deposits'], bins=50, color='lightgreen', edgecolor='black')
            ax.axvline(df['Total_Deposits'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: ${df["Total_Deposits"].mean():,.0f}')
            ax.set_xlabel('Deposit Amount ($)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Total Deposits', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Box plots
        st.markdown("#### Box Plots for Outlier Detection")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            bp = ax.boxplot(df['Total_Withdrawals'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightcoral')
            ax.set_title('Box Plot - Total Withdrawals', fontsize=14, fontweight='bold')
            ax.set_ylabel('Withdrawal Amount ($)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            bp = ax.boxplot(df['Total_Deposits'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            ax.set_title('Box Plot - Total Deposits', fontsize=14, fontweight='bold')
            ax.set_ylabel('Deposit Amount ($)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    # Tab 2: Time Trends
    with tab2:
        st.markdown('<h3 class="sub-header">Time-Based Trends</h3>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Withdrawals by Day of Week")
            fig, ax = plt.subplots(figsize=(10, 6))
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_withdrawals = df.groupby('Day_of_Week')['Total_Withdrawals'].mean().reindex(dow_order)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
            bars = ax.bar(dow_order, dow_withdrawals.values, color=colors, edgecolor='black')
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
            ax.set_title('Average Withdrawals by Day of Week', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
            
            # Best day insight
            best_day = dow_withdrawals.idxmax()
            st.info(f"🏆 **Highest withdrawal day:** {best_day} (${dow_withdrawals[best_day]:,.2f})")
        
        with col2:
            st.markdown("#### Withdrawals by Time of Day")
            fig, ax = plt.subplots(figsize=(10, 6))
            tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
            tod_withdrawals = df.groupby('Time_of_Day')['Total_Withdrawals'].mean().reindex(tod_order)
            bars = ax.bar(tod_order, tod_withdrawals.values, 
                         color=['#FFD93D', '#6BCB77', '#4D96FF', '#FF6B6B'], edgecolor='black')
            ax.set_xlabel('Time of Day', fontsize=12)
            ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
            ax.set_title('Average Withdrawals by Time of Day', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
            
            best_time = tod_withdrawals.idxmax()
            st.info(f"⏰ **Peak withdrawal time:** {best_time} (${tod_withdrawals[best_time]:,.2f})")
        
        # Monthly trend
        st.markdown("#### Monthly Withdrawal Trend")
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_withdrawals = df.groupby('Month')['Total_Withdrawals'].mean()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(monthly_withdrawals)), monthly_withdrawals.values, 
               color='steelblue', edgecolor='black')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
        ax.set_title('Monthly Withdrawal Trend', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(monthly_withdrawals)))
        ax.set_xticklabels([str(m) for m in monthly_withdrawals.index], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
    
    # Tab 3: Holiday & Events
    with tab3:
        st.markdown('<h3 class="sub-header">Holiday & Event Impact</h3>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Normal vs Holiday Days")
            fig, ax = plt.subplots(figsize=(8, 6))
            holiday_withdrawals = df.groupby('Holiday_Flag')['Total_Withdrawals'].mean()
            holiday_labels = ['Normal Day', 'Holiday']
            bars = ax.bar(holiday_labels, holiday_withdrawals.values, 
                         color=['#95E1D3', '#F38181'], edgecolor='black')
            ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
            ax.set_title('Withdrawals: Normal vs Holiday Days', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, holiday_withdrawals.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                       f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Special Event Impact")
            fig, ax = plt.subplots(figsize=(8, 6))
            event_withdrawals = df.groupby('Special_Event_Flag')['Total_Withdrawals'].mean()
            event_labels = ['No Event', 'Special Event']
            bars = ax.bar(event_labels, event_withdrawals.values, 
                         color=['#A8D8EA', '#AA96DA'], edgecolor='black')
            ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
            ax.set_title('Withdrawals: Normal vs Special Event Days', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, event_withdrawals.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                       f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
        
        # Combined analysis
        st.markdown("#### Combined Holiday & Event Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        combined = df.groupby(['Holiday_Flag', 'Special_Event_Flag'])['Total_Withdrawals'].mean().unstack()
        combined.plot(kind='bar', ax=ax, color=['#A8D8EA', '#AA96DA'], edgecolor='black')
        ax.set_xlabel('Holiday Flag', fontsize=12)
        ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
        ax.set_title('Withdrawals by Holiday and Event Combination', fontsize=14, fontweight='bold')
        ax.set_xticklabels(['Normal Day', 'Holiday'], rotation=0)
        ax.legend(['No Event', 'Special Event'])
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
        
        # Statistics
        holiday_change = ((df[df['Holiday_Flag']==1]['Total_Withdrawals'].mean() - 
                          df[df['Holiday_Flag']==0]['Total_Withdrawals'].mean()) / 
                         df[df['Holiday_Flag']==0]['Total_Withdrawals'].mean()) * 100
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>💡 Key Findings</h4>
            <ul>
                <li>Holiday days show a <b>{abs(holiday_change):.1f}% {'increase' if holiday_change > 0 else 'decrease'}</b> in withdrawals</li>
                <li>Total holiday records: <b>{df['Holiday_Flag'].sum()}</b></li>
                <li>Total special event records: <b>{df['Special_Event_Flag'].sum()}</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Tab 4: External Factors
    with tab4:
        st.markdown('<h3 class="sub-header">External Factors Analysis</h3>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Weather Impact")
            fig, ax = plt.subplots(figsize=(10, 6))
            weather_order = ['Clear', 'Cloudy', 'Rainy', 'Snowy']
            weather_withdrawals = df.groupby('Weather_Condition')['Total_Withdrawals'].mean().reindex(weather_order)
            weather_colors = ['#FFD93D', '#B0BEC5', '#64B5F6', '#E8EAF6']
            bars = ax.bar(weather_order, weather_withdrawals.values, color=weather_colors, edgecolor='black')
            ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
            ax.set_title('Average Withdrawals by Weather Condition', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
            
            st.info(f"🌤️ **Best weather:** {weather_withdrawals.idxmax()} (${weather_withdrawals.max():,.2f})")
        
        with col2:
            st.markdown("#### Location Type Impact")
            fig, ax = plt.subplots(figsize=(10, 6))
            location_withdrawals = df.groupby('Location_Type')['Total_Withdrawals'].mean().sort_values(ascending=False)
            location_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            bars = ax.bar(range(len(location_withdrawals)), location_withdrawals.values, 
                         color=location_colors[:len(location_withdrawals)], edgecolor='black')
            ax.set_xlabel('Location Type', fontsize=12)
            ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
            ax.set_title('Withdrawals by Location Type', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(location_withdrawals)))
            ax.set_xticklabels(location_withdrawals.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
            
            st.info(f"📍 **Best location:** {location_withdrawals.idxmax()} (${location_withdrawals.max():,.2f})")
        
        # Competitor ATM analysis
        st.markdown("#### Competitor ATM Impact")
        fig, ax = plt.subplots(figsize=(10, 6))
        competitor_withdrawals = df.groupby('Nearby_Competitor_ATMs')['Total_Withdrawals'].mean()
        ax.bar(competitor_withdrawals.index, competitor_withdrawals.values, 
               color='steelblue', edgecolor='black')
        ax.set_xlabel('Number of Nearby Competitor ATMs', fontsize=12)
        ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
        ax.set_title('Withdrawals by Number of Competitor ATMs', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
    
    # Tab 5: Relationships
    with tab5:
        st.markdown('<h3 class="sub-header">Relationship Analysis</h3>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cash Level vs Next Day Demand")
            sample_df = df.sample(n=min(1000, len(df)), random_state=42)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(sample_df['Previous_Day_Cash_Level'], 
                      sample_df['Cash_Demand_Next_Day'],
                      alpha=0.5, color='royalblue', s=20)
            
            # Trend line
            z = np.polyfit(sample_df['Previous_Day_Cash_Level'], 
                          sample_df['Cash_Demand_Next_Day'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sample_df['Previous_Day_Cash_Level'].min(), 
                                sample_df['Previous_Day_Cash_Level'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend Line')
            
            ax.set_xlabel('Previous Day Cash Level ($)', fontsize=12)
            ax.set_ylabel('Cash Demand Next Day ($)', fontsize=12)
            ax.set_title('Previous Day Cash Level vs Next Day Demand', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Correlation Heatmap")
            numeric_cols = ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 
                           'Cash_Demand_Next_Day', 'Holiday_Flag', 'Special_Event_Flag', 
                           'Nearby_Competitor_ATMs']
            correlation_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(numeric_cols, fontsize=9)
            ax.set_title('Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold')
            
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()

# =============================================================================
# CLUSTERING PAGE
# =============================================================================
def show_clustering_page(df):
    """Display the Clustering Analysis page"""
    st.markdown('<h1 class="main-header">🎯 ATM Clustering Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <h4>About K-Means Clustering</h4>
        <p>K-Means clustering groups ATMs based on similar demand patterns, enabling targeted cash management strategies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parameters
    st.markdown('<h3 class="sub-header">⚙️ Clustering Parameters</h3>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)
    
    with col2:
        random_state = st.number_input("Random State", min_value=0, max_value=100, value=42)
    
    with col3:
        run_clustering = st.button("🔄 Run Clustering", type="primary")
    
    # Prepare data for clustering
    st.markdown('<h3 class="sub-header">📊 ATM Feature Aggregation</h3>', 
                unsafe_allow_html=True)
    
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
    
    st.dataframe(atm_features.head(10), use_container_width=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    atm_features['Location_Encoded'] = le.fit_transform(atm_features['Location_Type'])
    
    clustering_features = ['Avg_Withdrawals', 'Std_Withdrawals', 'Max_Withdrawals',
                          'Avg_Deposits', 'Location_Encoded', 'Avg_Competitors']
    
    X = atm_features[clustering_features].copy()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal K
    st.markdown('<h3 class="sub-header">📈 Optimal K Determination</h3>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Elbow Method")
        wcss = []
        k_range = range(2, min(11, len(atm_features)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(list(k_range), wcss, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('WCSS', fontsize=12)
        ax.set_title('Elbow Method - Optimal K', fontsize=14, fontweight='bold')
        ax.axvline(x=n_clusters, color='red', linestyle='--', linewidth=2, 
                   label=f'Selected K = {n_clusters}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Silhouette Scores")
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Silhouette Scores', fontsize=14, fontweight='bold')
        ax.axvline(x=n_clusters, color='red', linestyle='--', linewidth=2,
                   label=f'Selected K = {n_clusters}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Apply K-Means
    st.markdown('<h3 class="sub-header">🎯 K-Means Clustering Results</h3>', 
                unsafe_allow_html=True)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    atm_features['Cluster'] = clusters
    
    # Assign labels
    cluster_labels = {}
    overall_avg = atm_features['Avg_Withdrawals'].mean()
    
    for cluster in range(n_clusters):
        cluster_data = atm_features[atm_features['Cluster'] == cluster]
        avg_withdrawal = cluster_data['Avg_Withdrawals'].mean()
        
        if avg_withdrawal > overall_avg * 1.2:
            cluster_labels[cluster] = "High-Demand"
        elif avg_withdrawal > overall_avg * 0.8:
            cluster_labels[cluster] = "Steady-Demand"
        else:
            cluster_labels[cluster] = "Low-Demand"
    
    atm_features['Cluster_Label'] = atm_features['Cluster'].map(cluster_labels)
    
    # Cluster summary
    st.markdown("#### Cluster Summary")
    
    cluster_summary = atm_features.groupby('Cluster').agg({
        'Avg_Withdrawals': 'mean',
        'Std_Withdrawals': 'mean',
        'ATM_ID': 'count',
        'Cluster_Label': 'first'
    }).reset_index()
    cluster_summary.columns = ['Cluster', 'Avg Withdrawal', 'Volatility', 'ATM Count', 'Label']
    
    st.dataframe(cluster_summary.style.format({
        'Avg Withdrawal': '${:,.2f}',
        'Volatility': '${:,.2f}'
    }), use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Clusters: Withdrawals vs Volatility")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                  '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8B500', '#82E0AA']
        
        for cluster in range(n_clusters):
            mask = atm_features['Cluster'] == cluster
            ax.scatter(atm_features.loc[mask, 'Avg_Withdrawals'],
                      atm_features.loc[mask, 'Std_Withdrawals'],
                      label=f'{cluster_labels[cluster]} (Cluster {cluster})',
                      s=100, alpha=0.7, color=colors[cluster], edgecolors='black')
        
        ax.set_xlabel('Average Withdrawals ($)', fontsize=12)
        ax.set_ylabel('Withdrawal Std Dev ($)', fontsize=12)
        ax.set_title('ATM Clusters: Avg Withdrawals vs Volatility', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### ATM Count by Cluster")
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_counts = atm_features['Cluster_Label'].value_counts()
        ax.bar(cluster_counts.index, cluster_counts.values, 
               color=colors[:len(cluster_counts)], edgecolor='black')
        ax.set_xlabel('Cluster Label', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Number of ATMs per Cluster', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
    
    # Location distribution by cluster
    st.markdown("#### Location Distribution by Cluster")
    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_location = atm_features.groupby(['Cluster_Label', 'Location_Type']).size().unstack(fill_value=0)
    cluster_location.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', edgecolor='black')
    ax.set_xlabel('Cluster Label', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Location Type Distribution by Cluster', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    plt.close()
    
    # ATM assignments
    st.markdown('<h3 class="sub-header">📋 ATM Cluster Assignments</h3>', 
                unsafe_allow_html=True)
    
    atm_assignments = atm_features[['ATM_ID', 'Cluster', 'Cluster_Label', 
                                    'Avg_Withdrawals', 'Std_Withdrawals', 'Location_Type']]
    st.dataframe(atm_assignments.sort_values('Cluster'), use_container_width=True)

# =============================================================================
# ANOMALY DETECTION PAGE
# =============================================================================
def show_anomaly_page(df):
    """Display the Anomaly Detection page"""
    st.markdown('<h1 class="main-header">🔍 Anomaly Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <h4>About Anomaly Detection</h4>
        <p>Anomalies are unusual or unexpected withdrawal patterns that deviate significantly from normal behavior. 
        Detecting them helps prevent cash shortages or excessive stock.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parameters
    st.markdown('<h3 class="sub-header">⚙️ Detection Parameters</h3>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        z_threshold = st.slider("Z-Score Threshold", min_value=2.0, max_value=5.0, value=3.0, step=0.5)
    
    with col2:
        contamination = st.slider("Isolation Forest Contamination", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    
    with col3:
        run_detection = st.button("🔄 Run Detection", type="primary")
    
    # Holiday vs Normal comparison
    st.markdown('<h3 class="sub-header">📊 Holiday vs Normal Day Comparison</h3>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    normal_withdrawals = df[df['Holiday_Flag'] == 0]['Total_Withdrawals']
    holiday_withdrawals = df[df['Holiday_Flag'] == 1]['Total_Withdrawals']
    
    with col1:
        st.metric("Normal Days", 
                  value=f"${normal_withdrawals.mean():,.2f}",
                  delta=f"{len(normal_withdrawals)} records")
    
    with col2:
        st.metric("Holiday Days", 
                  value=f"${holiday_withdrawals.mean():,.2f}",
                  delta=f"{len(holiday_withdrawals)} records")
    
    # Z-Score Detection
    st.markdown('<h3 class="sub-header">📈 Z-Score Based Detection</h3>', 
                unsafe_allow_html=True)
    
    df['Withdrawal_ZScore'] = (df['Total_Withdrawals'] - df['Total_Withdrawals'].mean()) / \
                              df['Total_Withdrawals'].std()
    df['Is_Anomaly_ZScore'] = np.abs(df['Withdrawal_ZScore']) > z_threshold
    
    num_anomalies_zscore = df['Is_Anomaly_ZScore'].sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Anomalies Detected (Z-Score)", 
                  value=f"{num_anomalies_zscore}",
                  delta=f"{num_anomalies_zscore/len(df)*100:.2f}% of total")
        
        st.markdown("#### Z-Score Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['Withdrawal_ZScore'], bins=50, color='lightblue', edgecolor='black')
        ax.axvline(z_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold = +{z_threshold}')
        ax.axvline(-z_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold = -{z_threshold}')
        ax.set_xlabel('Z-Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Z-Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Withdrawals vs Deposits (Anomalies Highlighted)")
        fig, ax = plt.subplots(figsize=(10, 6))
        normal_mask = ~df['Is_Anomaly_ZScore']
        ax.scatter(df.loc[normal_mask, 'Total_Withdrawals'],
                  df.loc[normal_mask, 'Total_Deposits'],
                  color='steelblue', alpha=0.3, s=20, label='Normal')
        ax.scatter(df.loc[~normal_mask, 'Total_Withdrawals'],
                  df.loc[~normal_mask, 'Total_Deposits'],
                  color='red', alpha=0.7, s=50, label='Anomaly', edgecolors='black')
        ax.set_xlabel('Total Withdrawals ($)', fontsize=12)
        ax.set_ylabel('Total Deposits ($)', fontsize=12)
        ax.set_title('Anomalies: Withdrawals vs Deposits', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Isolation Forest
    st.markdown('<h3 class="sub-header">🌲 Isolation Forest Detection</h3>', 
                unsafe_allow_html=True)
    
    isolation_features = ['Total_Withdrawals', 'Total_Deposits', 'Holiday_Flag',
                         'Special_Event_Flag', 'Previous_Day_Cash_Level']
    X_iso = df[isolation_features].copy()
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df['Is_Anomaly_IsoForest'] = iso_forest.fit_predict(X_iso)
    df['Is_Anomaly_IsoForest'] = df['Is_Anomaly_IsoForest'] == -1
    
    num_anomalies_iso = df['Is_Anomaly_IsoForest'].sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Anomalies Detected (Isolation Forest)", 
                  value=f"{num_anomalies_iso}",
                  delta=f"{num_anomalies_iso/len(df)*100:.2f}% of total")
    
    with col2:
        st.markdown("#### Anomaly Rate by Holiday/Event")
        fig, ax = plt.subplots(figsize=(10, 6))
        anomaly_by_flag = df.groupby(['Holiday_Flag', 'Special_Event_Flag'])['Is_Anomaly_ZScore'].mean() * 100
        anomaly_by_flag = anomaly_by_flag.unstack()
        anomaly_by_flag.plot(kind='bar', ax=ax, colormap='RdYlGn', edgecolor='black')
        ax.set_ylabel('Anomaly Rate (%)', fontsize=12)
        ax.set_xlabel('Holiday Flag', fontsize=12)
        ax.set_title('Anomaly Rate by Holiday/Event Flags', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(['No Event', 'Special Event'])
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
    
    # Anomaly Details
    st.markdown('<h3 class="sub-header">📋 Detected Anomalies</h3>', 
                unsafe_allow_html=True)
    
    anomalies_df = df[df['Is_Anomaly_ZScore']][['ATM_ID', 'Date', 'Day_of_Week', 'Time_of_Day',
                                                'Total_Withdrawals', 'Total_Deposits',
                                                'Holiday_Flag', 'Special_Event_Flag',
                                                'Weather_Condition', 'Withdrawal_ZScore']].copy()
    anomalies_df['Date'] = anomalies_df['Date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        anomalies_df.sort_values('Withdrawal_ZScore', ascending=False).style.format({
            'Total_Withdrawals': '${:,.0f}',
            'Total_Deposits': '${:,.0f}',
            'Withdrawal_ZScore': '{:.2f}'
        }),
        use_container_width=True
    )

# =============================================================================
# INTERACTIVE PLANNER PAGE
# =============================================================================
def show_planner_page(df):
    """Display the Interactive Planner page"""
    st.markdown('<h1 class="main-header">📈 Interactive Planner</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <h4>About Interactive Planner</h4>
        <p>Filter and explore ATM data dynamically to gain specific insights for cash management decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters
    st.markdown('<h3 class="sub-header">🔧 Filter Options</h3>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_day = st.multiselect(
            "Day of Week",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
    
    with col2:
        selected_time = st.multiselect(
            "Time of Day",
            options=['Morning', 'Afternoon', 'Evening', 'Night'],
            default=['Morning', 'Afternoon', 'Evening', 'Night']
        )
    
    with col3:
        selected_location = st.multiselect(
            "Location Type",
            options=df['Location_Type'].unique().tolist(),
            default=df['Location_Type'].unique().tolist()
        )
    
    with col4:
        selected_weather = st.multiselect(
            "Weather Condition",
            options=df['Weather_Condition'].unique().tolist(),
            default=df['Weather_Condition'].unique().tolist()
        )
    
    # Apply filters
    filtered_df = df[
        (df['Day_of_Week'].isin(selected_day)) &
        (df['Time_of_Day'].isin(selected_time)) &
        (df['Location_Type'].isin(selected_location)) &
        (df['Weather_Condition'].isin(selected_weather))
    ]
    
    # Display filtered stats
    st.markdown('<h3 class="sub-header">📊 Filtered Results</h3>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Records", value=f"{len(filtered_df):,}")
    
    with col2:
        st.metric("Avg Withdrawal", value=f"${filtered_df['Total_Withdrawals'].mean():,.0f}")
    
    with col3:
        st.metric("Max Withdrawal", value=f"${filtered_df['Total_Withdrawals'].max():,.0f}")
    
    with col4:
        st.metric("Total Volume", value=f"${filtered_df['Total_Withdrawals'].sum():,.0f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Withdrawal Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(filtered_df['Total_Withdrawals'], bins=30, color='skyblue', edgecolor='black')
        ax.axvline(filtered_df['Total_Withdrawals'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: ${filtered_df["Total_Withdrawals"].mean():,.0f}')
        ax.set_xlabel('Withdrawal Amount ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Withdrawals (Filtered)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### By Selected Categories")
        
        if len(selected_day) > 0:
            dow_avg = filtered_df.groupby('Day_of_Week')['Total_Withdrawals'].mean()
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_avg = dow_avg.reindex([d for d in dow_order if d in selected_day])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
            ax.bar(dow_avg.index, dow_avg.values, color=colors[:len(dow_avg)], edgecolor='black')
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Average Withdrawal ($)', fontsize=12)
            ax.set_title('Average Withdrawals by Day (Filtered)', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
    
    # Data preview
    st.markdown('<h3 class="sub-header">📋 Data Preview</h3>', 
                unsafe_allow_html=True)
    
    display_cols = ['ATM_ID', 'Date', 'Day_of_Week', 'Time_of_Day', 'Total_Withdrawals',
                   'Total_Deposits', 'Location_Type', 'Weather_Condition']
    
    st.dataframe(
        filtered_df[display_cols].head(100).style.format({
            'Total_Withdrawals': '${:,.0f}',
            'Total_Deposits': '${:,.0f}'
        }),
        use_container_width=True
    )
    
    # Download option
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_atm_data.csv',
        mime='text/csv'
    )

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main function to run the Streamlit app"""
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Dataset file 'atm_cash_management_dataset.csv' not found!")
        st.stop()
    
    # Create sidebar
    page, date_range, selected_atm, selected_location = create_sidebar()
    
    # Filter data based on sidebar selections
    filtered_df = filter_data(df, date_range, selected_atm, selected_location)
    
    # Show selected page
    if page == "🏠 Home":
        show_home_page(filtered_df)
    elif page == "📊 EDA":
        show_eda_page(filtered_df)
    elif page == "🎯 Clustering":
        show_clustering_page(filtered_df)
    elif page == "🔍 Anomaly Detection":
        show_anomaly_page(filtered_df)
    elif page == "📈 Interactive Planner":
        show_planner_page(filtered_df)
    
    # Display mascot/logo at bottom
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p style="color: #666; font-size: 12px;">
            ATM Intelligence Demand Forecasting | Formative Assessment 2
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load mascot image
    try:
        st.image("mascot.png", width=500, caption="FinTrust Bank Mascot")
    except:
        # If no mascot image, display a placeholder
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <span style="font-size: 50px;">🏦</span>
            <p style="color: #666; font-size: 12px;">FinTrust Bank Ltd.</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# RUN APP
# =============================================================================
if __name__ == "__main__":
    main()
