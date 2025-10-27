# Retail Sales Optimization - Capstone Project

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-Compatible-green.svg)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20DocumentDB-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Project Overview

This capstone project focuses on **Retail Sales Optimization** for ABC Retail Corp, implementing a comprehensive data engineering pipeline to ingest, clean, transform, and analyze large-scale retail datasets. The project generates actionable business insights through advanced analytics and machine learning models.

**Dataset Source:** [Customer Shopping Dataset - Kaggle](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset)

**Students** 
Nitin Bhoyate
Jai Kishore Rana
Bhaskar Adhikary
Vivek Kumar Srivastava
Vaibhav Darekar

## 🎯 Objectives

1. **Data Ingestion & Storage**
   - Load data from both AWS S3 and local filesystem
   - Store raw and processed data in MongoDB-compatible database
   - Support for multiple data formats (CSV, Excel, Parquet)

2. **Data Cleaning & Quality**
   - Handle missing values, outliers, and inconsistent formats
   - Maintain comprehensive data quality logs
   - Standardize dates, text, and categorical data

3. **Exploratory Data Analysis (EDA)**
   - Analyze sales patterns and seasonal trends
   - Evaluate product performance across categories
   - Understand customer behavior and demographics

4. **Feature Engineering & Modeling**
   - Create time-series features (lags, rolling windows)
   - Engineer customer-level features (RFM analysis)
   - Build predictive models for sales forecasting

5. **Data Warehousing & Optimization**
   - Design optimized MongoDB schema with indexing
   - Implement aggregation pipelines for fast queries
   - Export curated datasets in Parquet format

6. **Visualization & Reporting**
   - Interactive Streamlit dashboard
   - Comprehensive reports and presentations
   - Real-time business insights

## 🏗️ Project Structure

```
Retail-Sales-Optimization-Group-9/
│
├── config/                          # Configuration files
│   └── config.yaml                  # Main configuration
│
├── scripts/                         # Core Python scripts
│   ├── utils/                       # Utility modules
│   │   ├── __init__.py
│   │   ├── config_loader.py        # Config & client initialization
│   │   ├── logger.py               # Logging utilities
│   │   └── db_operations.py        # MongoDB operations
│   │
│   ├── data_ingestion/             # Data loading scripts
│   │   ├── __init__.py
│   │   ├── load_from_s3.py         # S3 data loader
│   │   ├── load_from_local.py      # Local data loader
│   │   └── data_loader.py          # Unified data loader
│   │
│   ├── data_cleaning/              # Data cleaning pipeline
│   │   ├── __init__.py
│   │   └── cleaning_pipeline.py    # Cleaning & preprocessing
│   │
│   └── etl/                        # ETL & transformation
│       ├── __init__.py
│       ├── feature_engineering.py  # Feature creation
│       ├── transformation_pipeline.py
│       └── aggregation.py          # MongoDB aggregations
│
├── notebooks/                       # Jupyter notebooks
│   ├── eda/
│   │   └── EDA_sales.ipynb         # Exploratory analysis
│   └── modeling/
│       └── modeling_forecasting.ipynb  # ML models
│
├── dashboards/                      # Visualization dashboards
│   └── app.py                      # Streamlit dashboard
│
├── docs/                           # Documentation
│   ├── SCHEMA_DESIGN.md           # MongoDB schema design
│   └── USER_GUIDE.md              # User guide
│
├── dataset/                        # Local data storage
│   ├── raw/
│   ├── cleaned/
│   └── transformed/
│
├── logs/                           # Application logs
│
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore
└── README.md

```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- MongoDB (local or AWS DocumentDB)
- AWS Account (optional, for S3 storage)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Retail-Sales-Optimization-Group-9
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Download the dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset)
   - Place the `shopping.csv` file in `dataset/raw/`

### Configuration

Edit `config/config.yaml` to customize:
- MongoDB connection settings
- AWS S3 bucket names
- Data cleaning parameters
- Feature engineering settings
- Model configurations

## 📊 Usage

### 1. Data Ingestion

**Load from local file:**
```bash
cd scripts/data_ingestion
python data_loader.py --source local --path ../../dataset/customer_shopping_data.csv --collection raw_sales --kaggle-dataset
```

**Load from S3:**
```bash
python data_loader.py --source s3 --path raw_data/customer_shopping_data.csv --collection raw_sales
```

### 2. Data Cleaning

```bash
cd scripts/data_cleaning
python cleaning_pipeline.py --input-collection raw_sales --output-collection cleaned_sales --shopping-dataset
```

### 3. Data Transformation

```bash
cd scripts/etl
python transformation_pipeline.py --input-collection cleaned_sales --output-collection transformed_sales --create-aggregations
```

### 4. Run Jupyter Notebooks

```bash
jupyter notebook notebooks/eda/EDA_sales.ipynb
jupyter notebook notebooks/modeling/modeling_forecasting.ipynb
```

### 5. Launch Dashboard

```bash
cd dashboards
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Run single click Pipeline
```bash
python run_pipeline.py
```

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.9+**: Primary programming language
- **MongoDB**: NoSQL database for flexible data storage
- **AWS S3**: Object storage for raw and processed data
- **AWS DocumentDB**: Managed MongoDB-compatible database (optional)

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **statsmodels**: Statistical models and time series
- **Prophet**: Time series forecasting

### Visualization
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations
- **Streamlit**: Web-based dashboards

### AWS & Cloud
- **boto3**: AWS SDK for Python
- **pymongo**: MongoDB driver for Python

### Development Tools
- **Jupyter**: Interactive notebooks
- **pytest**: Testing framework
- **python-dotenv**: Environment variable management

## 📈 Key Features

### Data Ingestion
- ✅ Dual-mode loading (S3 + Local)
- ✅ Support for CSV, Excel, Parquet
- ✅ Batch processing capabilities
- ✅ Error handling and logging

### Data Cleaning
- ✅ Automated missing value imputation
- ✅ Outlier detection and handling
- ✅ Date standardization
- ✅ Text normalization
- ✅ Duplicate removal
- ✅ Comprehensive cleaning logs

### Feature Engineering
- ✅ Time-based features (lag, rolling windows)
- ✅ Customer features (RFM analysis)
- ✅ Product/category features
- ✅ Aggregate statistics
- ✅ Categorical encoding

### Machine Learning
- ✅ Linear Regression
- ✅ Random Forest Regressor
- ✅ XGBoost (configurable)
- ✅ ARIMA/Prophet for time series
- ✅ Model evaluation metrics (RMSE, MAE, R²)

### Dashboard Features
- ✅ Real-time KPI monitoring
- ✅ Interactive sales trends
- ✅ Category performance analysis
- ✅ Customer demographics insights
- ✅ Shopping mall comparison
- ✅ Time-based analysis
- ✅ Data export functionality

## 📊 Dataset Information

**Source:** [Customer Shopping Dataset - Kaggle](https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset)

**Description:** The dataset contains shopping information from 10 different shopping malls in Istanbul, Turkey between 2021 and 2023.

**Columns:**
- `invoice_no`: Invoice number
- `customer_id`: Customer ID
- `gender`: Customer gender
- `age`: Customer age
- `category`: Product category
- `quantity`: Number of items purchased
- `price`: Price per item
- `payment_method`: Payment method used
- `invoice_date`: Date of purchase
- `shopping_mall`: Shopping mall location

**Size:** ~100,000+ transactions

## 🔍 MongoDB Schema Design

See [docs/SCHEMA_DESIGN.md](docs/SCHEMA_DESIGN.md) for detailed schema documentation.

### Collections:
- `raw_sales`: Original raw data
- `cleaned_sales`: Cleaned and standardized data
- `transformed_sales`: Feature-engineered data
- `aggregated_sales`: Pre-aggregated metrics
- `agg_daily_sales`: Daily aggregations
- `agg_category_sales`: Category-level metrics
- `agg_customer_summary`: Customer-level summaries

## 📝 Documentation

- **[Schema Design](docs/SCHEMA_DESIGN.md)**: MongoDB collection schemas
- **[User Guide](docs/USER_GUIDE.md)**: Comprehensive usage guide
- **Configuration**: See `config/config.yaml` for all settings

## 🧪 Testing

Run tests (when implemented):
```bash
pytest tests/
```

## 📈 Performance Optimization

### MongoDB Indexing
```python
# Create indexes for faster queries
db_handler.create_index('cleaned_sales', [('invoice_date', 1)])
db_handler.create_index('cleaned_sales', [('customer_id', 1)])
db_handler.create_index('cleaned_sales', [('category', 1)])
```

### Aggregation Pipeline
Utilize MongoDB's aggregation framework for efficient data processing.

### Parquet Export
Export large datasets to Parquet format for:
- 50-80% compression ratio
- Columnar storage benefits
- Fast analytical queries

## 🤝 Contributing

This is an academic capstone project. For contributions or suggestions:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

For questions or support, please contact the project team.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Mehmet Tahir Aslan - Kaggle
- **Institution**: IIT Jodhpur - M.Tech Data Science
- **Course**: Capstone Project
- **Academic Year**: 2025

## 📚 References

1. Customer Shopping Dataset - Kaggle: https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset
2. MongoDB Documentation: https://docs.mongodb.com/
3. AWS S3 Documentation: https://docs.aws.amazon.com/s3/
4. Streamlit Documentation: https://docs.streamlit.io/
5. scikit-learn Documentation: https://scikit-learn.org/

---

**Built with ❤️ for Retail Sales Optimization**

