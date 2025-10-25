"""
Streamlit Dashboard for Retail Sales Optimization
"""

import sys
from pathlib import Path

# Add scripts directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'scripts'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import hashlib

from utils.db_operations import MongoDBHandler
from utils.config_loader import load_config
from etl.aggregation import DataAggregator

# Page configuration
st.set_page_config(
    page_title="Retail Sales Optimization Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp > div:first-child {
        padding-top: 1rem;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .login-container {
        max-width: 350px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    .login-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .project-title {
        text-align: center;
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .group-title {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .error-message {
        color: #dc3545;
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def check_credentials(username, password):
    """Check if username and password are correct"""
    # Hash the password and compare
    hashed_password = hash_password(password)
    correct_username = "admin"
    correct_password_hash = hash_password("admin")
    
    return username == correct_username and hashed_password == correct_password_hash


def show_login_screen():
    """Display login screen"""
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="project-title">GROUP 9 PROJECT</h1>', unsafe_allow_html=True)
        st.markdown('<h2 class="group-title">RETAIL SALES OPTIMIZATION</h2>', unsafe_allow_html=True)
        st.markdown('<h3 class="login-title">🔐 Login to Dashboard</h3>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if check_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add some info about the demo
        st.markdown("---")
        st.info("**Demo Credentials:** Username: `admin`, Password: `admin`")


def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()


@st.cache_data
def load_data(collection_name='cleaned_sales'):
    """Load data from MongoDB with caching"""
    db_handler = MongoDBHandler()
    df = db_handler.read_to_dataframe(collection_name)
    
    # Ensure date is datetime
    if 'invoice_date' in df.columns:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    
    return df


@st.cache_data
def get_aggregated_data(collection_name='cleaned_sales'):
    """Get pre-aggregated data"""
    aggregator = DataAggregator()
    
    # Get various aggregations
    category_data = aggregator.aggregate_by_category(collection_name)
    demographic_data = aggregator.aggregate_by_demographics(collection_name)
    payment_data = aggregator.aggregate_payment_methods(collection_name)
    
    return {
        'category': pd.DataFrame(category_data) if category_data else None,
        'demographics': demographic_data,
        'payment': pd.DataFrame(payment_data) if payment_data else None
    }


def main():
    """Main dashboard function"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Check authentication
    if not st.session_state.authenticated:
        show_login_screen()
        return
    
    # Header with logout button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<p class="main-header">🛍️ Retail Sales Optimization Dashboard</p>', unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Logout", key="logout_btn"):
            logout()
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    st.sidebar.markdown(f"**Welcome, {st.session_state.username}!**")
    st.sidebar.markdown("---")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["cleaned_sales", "transformed_sales"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data(data_source)
    
    # Date range filter
    if 'invoice_date' in df.columns:
        min_date = df['invoice_date'].min().date()
        max_date = df['invoice_date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            df = df[(df['invoice_date'].dt.date >= date_range[0]) & 
                   (df['invoice_date'].dt.date <= date_range[1])]
    
    # Category filter
    if 'category' in df.columns:
        categories = ['All'] + sorted(df['category'].unique().tolist())
        selected_category = st.sidebar.selectbox("Select Category", categories)
        
        if selected_category != 'All':
            df = df[df['category'] == selected_category]
    
    st.sidebar.markdown("---")
    st.sidebar.info("Dataset: Customer Shopping Dataset from Kaggle")
    
    # Main content
    if df.empty:
        st.error("No data available for the selected filters.")
        return
    
    # KPI Metrics
    st.header("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_sales = df['total_amount'].sum()
        st.metric("Total Sales", f"${total_sales:,.2f}")
    
    with col2:
        avg_transaction = df['total_amount'].mean()
        st.metric("Avg Transaction", f"${avg_transaction:,.2f}")
    
    with col3:
        total_transactions = len(df)
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col4:
        unique_customers = df['customer_id'].nunique()
        st.metric("Unique Customers", f"{unique_customers:,}")
    
    with col5:
        avg_items = df['quantity'].mean()
        st.metric("Avg Items/Transaction", f"{avg_items:.2f}")
    
    st.markdown("---")
    
    # Sales Trends
    st.header("📈 Sales Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily sales trend
        daily_sales = df.groupby('invoice_date')['total_amount'].sum().reset_index()
        
        fig = px.line(
            daily_sales,
            x='invoice_date',
            y='total_amount',
            title='Daily Sales Trend',
            labels={'invoice_date': 'Date', 'total_amount': 'Total Sales'}
        )
        fig.update_traces(line_color='#1f77b4', line_width=2)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly sales
        df['month'] = df['invoice_date'].dt.to_period('M').astype(str)
        monthly_sales = df.groupby('month')['total_amount'].sum().reset_index()
        
        fig = px.bar(
            monthly_sales,
            x='month',
            y='total_amount',
            title='Monthly Sales',
            labels={'month': 'Month', 'total_amount': 'Total Sales'},
            color='total_amount',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Category Analysis
    st.header("🏷️ Category Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by category
        category_sales = df.groupby('category')['total_amount'].sum().sort_values(ascending=False).reset_index()
        
        fig = px.bar(
            category_sales,
            x='category',
            y='total_amount',
            title='Sales by Category',
            labels={'category': 'Category', 'total_amount': 'Total Sales'},
            color='total_amount',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category pie chart
        fig = px.pie(
            category_sales,
            values='total_amount',
            names='category',
            title='Sales Share by Category',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Customer Insights
    st.header("👥 Customer Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Gender distribution
        gender_sales = df.groupby('gender')['total_amount'].sum().reset_index()
        
        fig = px.bar(
            gender_sales,
            x='gender',
            y='total_amount',
            title='Sales by Gender',
            labels={'gender': 'Gender', 'total_amount': 'Total Sales'},
            color='gender',
            color_discrete_map={'Male': '#4ECDC4', 'Female': '#FF6B9D'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution
        fig = px.histogram(
            df,
            x='age',
            nbins=30,
            title='Customer Age Distribution',
            labels={'age': 'Age', 'count': 'Number of Customers'}
        )
        fig.update_traces(marker_color='lightgreen')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Payment methods
        payment_sales = df.groupby('payment_method')['total_amount'].sum().reset_index()
        
        fig = px.pie(
            payment_sales,
            values='total_amount',
            names='payment_method',
            title='Payment Method Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Shopping Mall Performance
    st.header("🏬 Shopping Mall Performance")
    
    mall_sales = df.groupby('shopping_mall').agg({
        'total_amount': 'sum',
        'invoice_no': 'count',
        'customer_id': 'nunique'
    }).reset_index().sort_values('total_amount', ascending=False)
    
    mall_sales.columns = ['Shopping Mall', 'Total Sales', 'Transactions', 'Unique Customers']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            mall_sales,
            x='Shopping Mall',
            y='Total Sales',
            title='Sales by Shopping Mall',
            color='Total Sales',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            mall_sales.style.format({
                'Total Sales': '${:,.2f}',
                'Transactions': '{:,}',
                'Unique Customers': '{:,}'
            }),
            height=400
        )
    
    st.markdown("---")
    
    # Time-based Analysis
    st.header("⏰ Time-based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week analysis
        df['day_of_week'] = df['invoice_date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        dow_sales = df.groupby('day_of_week')['total_amount'].sum().reindex(day_order).reset_index()
        
        fig = px.bar(
            dow_sales,
            x='day_of_week',
            y='total_amount',
            title='Sales by Day of Week',
            labels={'day_of_week': 'Day', 'total_amount': 'Total Sales'},
            color='total_amount',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hour of day (if available)
        if 'hour' in df.columns:
            hourly_sales = df.groupby('hour')['total_amount'].sum().reset_index()
            
            fig = px.line(
                hourly_sales,
                x='hour',
                y='total_amount',
                title='Sales by Hour of Day',
                labels={'hour': 'Hour', 'total_amount': 'Total Sales'}
            )
            fig.update_traces(line_color='coral', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Hourly data not available")
    
    st.markdown("---")
    
    # Top Performers
    st.header("🏆 Top Performers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Customers")
        top_customers = df.groupby('customer_id')['total_amount'].sum().sort_values(ascending=False).head(10).reset_index()
        top_customers.columns = ['Customer ID', 'Total Spent']
        st.dataframe(
            top_customers.style.format({'Total Spent': '${:,.2f}'}),
            height=400
        )
    
    with col2:
        st.subheader("Top 10 Categories")
        top_categories = df.groupby('category')['total_amount'].sum().sort_values(ascending=False).head(10).reset_index()
        top_categories.columns = ['Category', 'Total Sales']
        st.dataframe(
            top_categories.style.format({'Total Sales': '${:,.2f}'}),
            height=400
        )
    
    st.markdown("---")
    
    # Raw Data
    with st.expander("📋 View Raw Data"):
        st.dataframe(df.head(1000))
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"retail_sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            <p>Retail Sales Optimization Dashboard | Data from Kaggle Customer Shopping Dataset</p>
            <p>Built with Streamlit | © 2025</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()

