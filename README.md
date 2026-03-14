🏦 ATM Intelligence Demand Forecasting Dashboard

An interactive Streamlit dashboard designed to analyze ATM transaction data and support cash demand forecasting using data mining techniques.

This project was developed as part of Formative Assessment 2 – Interactive Data Mining Application. The application allows users to explore ATM transaction data, detect unusual patterns, and analyze ATM performance using clustering and anomaly detection techniques.

📊 Project Overview

Banks need to maintain the right amount of cash in ATMs.
Too little cash causes service disruption, while too much cash leads to inefficient resource allocation.

This dashboard helps financial institutions:

• Analyze ATM transaction patterns
• Identify high-demand ATMs
• Detect unusual withdrawal activity
• Support smarter cash management decisions

The application uses Exploratory Data Analysis, Machine Learning clustering, and anomaly detection to generate insights from ATM transaction data.

⚙️ Technologies Used

The project is implemented using the following technologies:

Technology	Purpose
Python	Core programming language
Streamlit	Interactive web dashboard
Pandas	Data manipulation and analysis
NumPy	Numerical computations
Matplotlib	Data visualization
Seaborn	Statistical visualization
Scikit-learn	Machine learning models

📁 Project Structure
ATM-Intelligence-Dashboard/
│
├── app.py
├── atm_cash_management_dataset.csv
├── mascot.png
├── README.md

Files Explanation

• app.py – Main Streamlit application
• atm_cash_management_dataset.csv – ATM transaction dataset
• mascot.png – Dashboard mascot/logo
• README.md – Project documentation

🚀 Features

The dashboard contains five main analytical sections.

🏠 Home Page

Provides an overview of ATM data including:

• Total records
• Number of ATMs
• Average withdrawals
• Total transaction volume
• Timeline of ATM withdrawals

This page gives users a quick summary of the entire dataset.

📊 Exploratory Data Analysis (EDA)

The EDA section helps users understand ATM usage patterns.

It includes:

• Withdrawal and deposit distributions
• Boxplots for detecting outliers
• Daily and monthly transaction trends
• Impact of holidays and special events
• Weather and location analysis
• Correlation heatmap between variables

These visualizations help identify patterns that influence ATM demand.

🎯 Clustering Analysis

The clustering module groups ATMs with similar demand patterns using K-Means clustering.

Steps performed:

Feature aggregation by ATM

Feature scaling using StandardScaler

Optimal cluster selection using:

• Elbow Method
• Silhouette Score

Clusters are categorized as:

• High-Demand ATMs
• Steady-Demand ATMs
• Low-Demand ATMs

This helps banks prioritize cash replenishment.

🔍 Anomaly Detection

Anomaly detection identifies unusual ATM withdrawal patterns.

Two techniques are used:

Z-Score Detection

Detects extreme values that deviate significantly from the mean.

Isolation Forest

A machine learning algorithm used for detecting anomalies in multidimensional data.

This helps detect:

• Unexpected spikes in withdrawals
• Unusual ATM behavior
• Potential system issues

📈 Interactive Planner

The interactive planner allows users to filter ATM data dynamically based on:

• Day of week
• Time of day
• Location type
• Weather conditions

Users can explore customized insights and download filtered datasets for further analysis.

🧠 Machine Learning Techniques Used
K-Means Clustering

Groups ATMs based on transaction patterns.

Isolation Forest

Detects anomalies in ATM transaction data.

Z-Score Analysis

Identifies statistical outliers.

📊 Key Insights Generated

The dashboard helps identify:

• Peak withdrawal days
• Best performing ATM locations
• Holiday impact on ATM usage
• High-demand ATMs requiring frequent cash replenishment
• Unusual withdrawal patterns

This project demonstrates how data mining and machine learning techniques can be applied to real-world banking problems such as ATM cash demand forecasting.

It integrates:

• Data analysis
• Visualization
• Machine learning
• Interactive dashboards

Project developed for Formative Assessment 2 – Interactive Data Mining Application.
