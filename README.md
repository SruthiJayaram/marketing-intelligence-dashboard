# Marketing Intelligence Dashboard 📊

A comprehensive marketing analytics platform that processes multi-channel advertising data and provides advanced business intelligence insights through an interactive Streamlit dashboard.

## 🎯 Overview

This project creates a unified marketing intelligence system that:
- Processes data from Facebook, Google, TikTok, and business operations
- Provides real-time KPI monitoring and trend analysis
- Delivers advanced analytics including forecasting, attribution modeling, and campaign optimization insights
- Enables drill-down analysis for tactical decision making

## ✨ Key Features

### 📈 Core Analytics
- **Multi-channel KPI Dashboard**: Revenue, ROAS, CTR, CPC metrics across all channels
- **Interactive Filtering**: Date range and channel selection for focused analysis
- **Daily Trend Visualization**: Track performance metrics over time
- **Channel Performance Comparison**: Side-by-side channel breakdown and rankings

### 🔬 Advanced Analytics
- **Revenue Forecasting**: ML-powered predictions with configurable forecast periods
- **Seasonality Analysis**: Day-of-week patterns and weekly trend identification
- **Attribution Modeling**: Multi-touch attribution with 7-day lag analysis
- **Campaign Drill-down**: Granular campaign and tactic performance analysis

### 📊 Business Intelligence
- **Correlation Analysis**: Marketing spend impact on business outcomes
- **Lag Effect Studies**: Delayed impact measurement of advertising spend
- **Performance Benchmarking**: Cross-channel efficiency comparisons
- **Tactical Insights**: Best performing strategies and optimization opportunities

## 🏗️ Project Structure

```
marketing-intelligence-dashboard/
├── 📁 app/
│   └── 📄 app.py                 # Main Streamlit dashboard application
├── 📁 data/                      # Data directory (source files only in repo)
│   ├── 📄 Facebook.csv           # Facebook advertising campaign data
│   ├── 📄 Google.csv            # Google Ads campaign data  
│   ├── 📄 TikTok.csv            # TikTok advertising campaign data
│   ├── 📄 Business.csv          # Business operations and revenue data
│   └── 📄 df_final.csv          # ⚠️ Generated file (not in repo)
├── 📁 notebooks/
│   └── 📄 01_data_prep.ipynb    # Data processing and ETL pipeline
├── 📄 requirements.txt          # Python package dependencies
├── 📄 README.md                # 📖 This documentation file
├── 📄 .gitignore               # Git exclusion rules
└── 📁 .git/                    # Git version control (hidden)
```

### File Descriptions

#### 📱 **Application Files**
- **`app/app.py`** - Complete Streamlit web application with:
  - Interactive dashboard interface
  - Advanced analytics engine (forecasting, attribution, seasonality)
  - Real-time data filtering and visualization
  - Campaign drill-down capabilities

#### 📊 **Data Files**
- **Source Data** (included in repository):
  - `data/Facebook.csv` - Raw Facebook campaign performance metrics
  - `data/Google.csv` - Raw Google Ads campaign data  
  - `data/TikTok.csv` - Raw TikTok advertising metrics
  - `data/Business.csv` - Daily business revenue and operations data

- **Generated Data** (created by notebook, excluded from repo):
  - `data/df_final.csv` - Processed, unified daily dataset for dashboard

#### 🔬 **Analysis Files**
- **`notebooks/01_data_prep.ipynb`** - Comprehensive data processing pipeline:
  - Loads and validates all source CSV files
  - Standardizes column names and data formats
  - Calculates marketing metrics (CTR, CPC, ROAS)
  - Aggregates data by date and channel
  - Exports clean dataset for dashboard consumption

#### ⚙️ **Configuration Files**
- **`requirements.txt`** - Python dependencies including:
  - `streamlit` - Web application framework
  - `pandas` - Data manipulation and analysis
  - `plotly` - Interactive visualization library
  - `scikit-learn` - Machine learning for forecasting
  - `numpy` - Numerical computing support

- **`.gitignore`** - Excludes from version control:
  - Generated data files (`df_final.csv`)
  - Python cache files (`__pycache__/`)
  - Virtual environments (`venv/`, `.env`)
  - IDE configuration files
  - Streamlit cache (`.streamlit/`)

### Repository Organization Principles

#### ✅ **Included in Git**
- Source code (`app/app.py`, `notebooks/*.ipynb`)
- Source data files (raw CSV files)
- Documentation (`README.md`)
- Dependency specifications (`requirements.txt`)
- Configuration files (`.gitignore`)

#### ❌ **Excluded from Git**
- Generated/processed data files
- Python bytecode and cache
- Virtual environment directories
- IDE-specific configuration
- User-specific settings

This structure ensures:
- **Reproducibility**: Anyone can regenerate processed data
- **Collaboration**: No conflicts from generated files
- **Security**: Sensitive cached data stays local
- **Performance**: Faster clone/pull operations

## 🚀 Complete Setup Guide

### Prerequisites
- **Python 3.8 or higher**
- **pip package manager**
- **Git** (for cloning the repository)
- **Jupyter Notebook or JupyterLab** (for data processing)

### Step-by-Step Installation

#### 1. **Clone the Repository**
```bash
git clone https://github.com/SruthiJayaram/marketing-intelligence-dashboard.git
cd marketing-intelligence-dashboard
```

#### 2. **Set Up Python Environment** (Recommended)
```bash
# Create a virtual environment
python -m venv marketing_dashboard_env

# Activate the environment
# On Windows:
marketing_dashboard_env\Scripts\activate
# On macOS/Linux:
source marketing_dashboard_env/bin/activate
```

#### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

#### 4. **Verify Data Files**
Ensure the following source data files are in the `data/` directory:
- `data/Facebook.csv` - Facebook advertising data
- `data/Google.csv` - Google Ads data  
- `data/TikTok.csv` - TikTok advertising data
- `data/Business.csv` - Business operations data

#### 5. **Process the Data** (Required First Step)
Before running the dashboard, you must process the raw data:

```bash
# Start Jupyter Notebook
jupyter notebook

# Open and run notebooks/01_data_prep.ipynb
# This will:
# - Load and clean all CSV files
# - Standardize column names
# - Calculate key metrics (CTR, CPC, ROAS)
# - Generate the processed dataset: data/df_final.csv
```

**⚠️ Important**: The notebook **must be run successfully** before launching the dashboard. It generates `df_final.csv` which the dashboard requires.

#### 6. **Launch the Dashboard**
```bash
streamlit run app/app.py
```

#### 7. **Access the Application**
- Open your web browser
- Navigate to `http://localhost:8501`
- Use the sidebar filters to explore different date ranges and channels

### 📁 Generated Files

The following files are automatically generated during setup:

| File | Generated By | Purpose | Include in Git? |
|------|-------------|---------|-----------------|
| `data/df_final.csv` | `01_data_prep.ipynb` | Processed unified dataset for dashboard | ❌ No (regenerated) |
| `.streamlit/` | Streamlit | App configuration cache | ❌ No (excluded) |
| `__pycache__/` | Python | Compiled bytecode | ❌ No (excluded) |

### 🔧 Troubleshooting

#### Common Issues:

**1. Dashboard shows "File not found" error:**
```bash
# Solution: Run the data processing notebook first
jupyter notebook notebooks/01_data_prep.ipynb
# Execute all cells to generate df_final.csv
```

**2. Missing dependencies error:**
```bash
# Solution: Install/update requirements
pip install -r requirements.txt --upgrade
```

**3. CSV column errors:**
```bash
# Solution: Verify source CSV files have correct structure
# Check data/ folder for required files:
ls -la data/
```

**4. Port already in use:**
```bash
# Solution: Use a different port
streamlit run app/app.py --server.port 8502
```

### 🔄 Development Workflow

#### Making Changes:
1. **Modify Data Processing**: Edit `notebooks/01_data_prep.ipynb`
2. **Update Dashboard**: Edit `app/app.py`
3. **Test Changes**: Restart Streamlit app
4. **Regenerate Data**: Re-run notebook if data logic changes

#### Adding New Data Sources:
1. Add new CSV files to `data/` directory
2. Update `notebooks/01_data_prep.ipynb` to include new sources
3. Modify `app/app.py` to handle new data fields
4. Update this README with new data requirements

## 📊 Data Requirements & File Structure

### Required Source Files

The system expects the following CSV files in the `data/` directory:

#### Marketing Channel Data
**Facebook.csv, Google.csv, TikTok.csv** must contain:
- `date` - Campaign date (YYYY-MM-DD format)
- `campaign` - Campaign name/identifier  
- `tactic` - Marketing tactic/strategy used
- `spend` - Advertising spend amount
- `impressions` (or `impression`) - Ad impressions count
- `clicks` - Ad clicks count  
- `revenue` (or `attributed_revenue`) - Revenue attributed to ads

#### Business Operations Data
**Business.csv** must contain:
- `date` - Business date (YYYY-MM-DD format)
- `total_revenue` - Daily total business revenue
- `orders` - Number of orders/transactions
- Additional business metrics (optional)

### Generated Files

#### Processed Dataset (df_final.csv)
Generated by running `notebooks/01_data_prep.ipynb`:
- **Purpose**: Unified daily dataset combining all channels and business data
- **Columns**: Standardized metrics across all data sources
- **Usage**: Primary data source for the dashboard
- **Generation**: Automatically created when notebook is executed

### Data Processing Pipeline

The `01_data_prep.ipynb` notebook performs:

1. **Data Loading**: Reads all source CSV files
2. **Column Standardization**: Harmonizes field names across sources
3. **Data Cleaning**: Handles missing values and data type conversions
4. **Metric Calculation**: Computes CTR, CPC, ROAS for each channel
5. **Data Aggregation**: Creates daily summaries by channel
6. **Export**: Saves processed data as `df_final.csv`

### Sample Data Structure

#### Source File Example (Facebook.csv):
```csv
date,campaign,tactic,spend,impressions,clicks,revenue
2024-01-01,Campaign_A,Video_Ads,1000,50000,2500,5000
2024-01-02,Campaign_B,Carousel_Ads,800,40000,1800,3600
```

#### Generated File (df_final.csv):
```csv
date,fb_spend,fb_revenue,fb_clicks,fb_impressions,gg_spend,gg_revenue,tt_spend,tt_revenue,total_spend,total_revenue,business_total_revenue,business_orders
2024-01-01,1000,5000,2500,50000,1200,4800,800,3200,3000,13000,15000,120
```

## 🎛️ Using the Dashboard

### Launching the Application

1. **Ensure Prerequisites**: 
   - All dependencies installed (`pip install -r requirements.txt`)
   - Data processed (`notebooks/01_data_prep.ipynb` executed successfully)
   - `df_final.csv` exists in `data/` directory

2. **Start the Dashboard**:
   ```bash
   cd marketing-intelligence-dashboard
   streamlit run app/app.py
   ```

3. **Access the Interface**:
   - **Local URL**: `http://localhost:8501`
   - **Network URL**: Available to other devices on your network
   - **External URL**: Public access (if configured)

### Dashboard Navigation

#### Sidebar Controls
- **Date Range Filter**: Select specific time periods for analysis
- **Channel Selection**: Choose which marketing channels to include
- **Real-time Updates**: All visualizations update automatically with filter changes

#### Main Dashboard Sections

##### 📊 **KPI Overview**
- **Total Revenue**: Aggregated revenue across all selected channels and dates
- **Gross Profit**: Business profitability metrics (when available)
- **Marketing Spend**: Total advertising investment
- **ROAS**: Return on advertising spend ratio
- **Orders**: Total transaction volume

##### 📈 **Daily Trends**
- Interactive line charts showing performance over time
- Configurable metrics: Revenue, Orders, Spend
- Hover details for specific data points

##### 🎯 **Channel Breakdown**
- Comparative analysis across Facebook, Google, TikTok
- Switch between Spend and Revenue views
- Color-coded channel performance

##### 📋 **Performance Metrics Table**
- Channel-specific KPIs: CTR, CPC, ROAS
- Formatted for easy comparison
- Real-time calculations based on filters

#### 🔬 **Advanced Analytics Tabs**

##### 📈 **Forecasting**
- **Revenue Predictions**: ML-powered forecasts using linear regression
- **Configurable Period**: Adjust forecast horizon (7-30 days)
- **Model Accuracy**: R² score showing prediction reliability
- **Visual Comparison**: Historical vs predicted data overlay

##### 📊 **Seasonality Analysis**
- **Day-of-Week Patterns**: Identify optimal days for campaigns
- **Weekly Trends**: Track performance evolution over weeks
- **Performance Optimization**: Data-driven timing recommendations

##### 🎯 **Attribution Modeling**
- **Channel Correlations**: Statistical relationships with business outcomes
- **Multi-Touch Attribution**: Understanding channel interactions
- **7-Day Attribution Windows**: Delayed impact analysis of ad spend
- **Lag Effect Visualization**: When advertising impact materializes

##### 🔍 **Campaign Drill-Down**
- **Channel-Specific Analysis**: Deep dive into individual platforms
- **Campaign Rankings**: Performance-sorted campaign lists
- **Tactical Insights**: Best performing strategies identification
- **Performance Distributions**: CTR, CPC, ROAS histograms
- **Strategic Recommendations**: Data-driven optimization suggestions

### Expected User Workflow

1. **Start with Overview**: Review high-level KPIs and trends
2. **Apply Filters**: Focus on specific time periods or channels
3. **Analyze Trends**: Identify patterns in daily performance
4. **Compare Channels**: Understand relative channel performance
5. **Dive into Advanced Analytics**: 
   - Use forecasting for budget planning
   - Analyze seasonality for campaign timing
   - Study attribution for budget allocation
   - Drill down to campaign level for tactical optimization

### Performance Notes

- **Data Caching**: Streamlit caches data loading for faster interactions
- **Real-time Updates**: All charts update automatically with filter changes
- **Interactive Visualizations**: Hover, zoom, and click for detailed insights
- **Responsive Design**: Works on desktop and tablet devices

## 🔧 Technical Implementation

### Data Processing Pipeline
- **ETL Process**: Automated data cleaning and standardization
- **Column Harmonization**: Consistent naming across data sources
- **Metric Calculations**: CTR, CPC, ROAS computation
- **Data Aggregation**: Channel and date-level summaries

### Analytics Engine
- **Forecasting**: scikit-learn LinearRegression
- **Statistical Analysis**: scipy.stats for correlation analysis
- **Visualization**: Plotly for interactive charts
- **Performance**: Streamlit caching for optimal load times

### Libraries Used
```python
streamlit>=1.28.0       # Web application framework
pandas>=1.5.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
plotly>=5.15.0         # Interactive visualizations
scikit-learn>=1.3.0    # Machine learning
scipy>=1.11.0          # Statistical analysis
statsmodels>=0.14.0    # Advanced statistics
```

## 🎯 Business Value

### For Marketing Teams
- **Campaign Optimization**: Identify top-performing campaigns and tactics
- **Budget Allocation**: Data-driven spend distribution across channels
- **Performance Monitoring**: Real-time KPI tracking and alerts

### For Business Intelligence
- **Forecasting**: Revenue predictions for planning and budgeting
- **Attribution**: True understanding of channel contribution
- **Seasonality**: Optimal timing for campaigns and promotions

### For Executives
- **ROI Visibility**: Clear ROAS metrics across all channels
- **Strategic Insights**: High-level trends and performance drivers
- **Decision Support**: Data-backed recommendations for growth

## 🔄 Data Flow

1. **Raw Data Ingestion**: CSV files from advertising platforms and business systems
2. **Data Processing**: Cleaning, standardization, and metric calculation
3. **Unified Dataset**: Single source of truth in `df_final.csv`
4. **Interactive Analysis**: Real-time filtering and visualization
5. **Advanced Analytics**: ML-powered insights and predictions

## 🚀 Future Enhancements

- **Real-time Data Integration**: API connections to live data sources
- **Advanced ML Models**: Prophet/ARIMA for improved forecasting
- **Automated Reporting**: Scheduled email reports and alerts
- **Custom Dashboards**: User-specific view configurations
- **A/B Testing Framework**: Experiment tracking and analysis

## 📝 License

This project is developed for educational and assessment purposes. Please refer to the repository settings for licensing information.

## 🤝 Contributing

This project was created as part of a marketing analytics assessment. For questions or suggestions, please contact the repository owner.

---

**Built with** ❤️ **using Python, Streamlit, and modern data science tools**
