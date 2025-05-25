"""
Streamlit Dashboard for Job Monitoring System.
Interactive web interface for viewing and filtering job data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_data
def load_data():
    """Load processed job data with caching."""
    data_path = "data/processed_jobs.csv"
    
    if not os.path.exists(data_path):
        return None
    
    try:
        df = pd.read_csv(data_path)
        
        # Convert date columns if they exist
        date_columns = ['date_posted', 'scraped_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_metrics_row(df):
    """Create metrics row at the top of dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jobs", len(df))
    
    with col2:
        unique_companies = df['company'].nunique()
        st.metric("Companies", unique_companies)
    
    with col3:
        unique_locations = df['location'].nunique()
        st.metric("Locations", unique_locations)
    
    with col4:
        if 'cluster' in df.columns:
            unique_clusters = df['cluster'].nunique()
            st.metric("Job Categories", unique_clusters)
        else:
            st.metric("Skills Identified", df.columns.str.startswith('skill_').sum())

def create_visualizations(df):
    """Create data visualizations."""
    st.subheader("📊 Data Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top companies chart
        st.subheader("Top Companies")
        top_companies = df['company'].value_counts().head(10)
        
        if not top_companies.empty:
            fig_companies = px.bar(
                x=top_companies.values,
                y=top_companies.index,
                orientation='h',
                title="Jobs by Company",
                labels={'x': 'Number of Jobs', 'y': 'Company'}
            )
            fig_companies.update_layout(height=400)
            st.plotly_chart(fig_companies, use_container_width=True)
        else:
            st.info("No company data available")
    
    with col2:
        # Top locations chart
        st.subheader("Top Locations")
        top_locations = df['location'].value_counts().head(10)
        
        if not top_locations.empty:
            fig_locations = px.bar(
                x=top_locations.values,
                y=top_locations.index,
                orientation='h',
                title="Jobs by Location",
                labels={'x': 'Number of Jobs', 'y': 'Location'}
            )
            fig_locations.update_layout(height=400)
            st.plotly_chart(fig_locations, use_container_width=True)
        else:
            st.info("No location data available")
    
    # Cluster distribution if available
    if 'cluster' in df.columns:
        st.subheader("Job Categories Distribution")
        
        cluster_counts = df['cluster'].value_counts().sort_index()
        cluster_labels = []
        
        # Try to get cluster labels if available
        for cluster_id in cluster_counts.index:
            if 'cluster_label' in df.columns:
                label = df[df['cluster'] == cluster_id]['cluster_label'].iloc[0]
                cluster_labels.append(f"Cluster {cluster_id}: {label[:50]}...")
            else:
                cluster_labels.append(f"Cluster {cluster_id}")
        
        fig_clusters = px.pie(
            values=cluster_counts.values,
            names=cluster_labels,
            title="Distribution of Job Categories"
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
    
    # Skills analysis if skill columns exist
    skill_columns = [col for col in df.columns if col.startswith('skill_')]
    if skill_columns:
        st.subheader("Top Skills Required")
        
        # Calculate skill frequencies
        skill_sums = df[skill_columns].sum().sort_values(ascending=False).head(15)
        skill_names = [col.replace('skill_', '').replace('_', ' ').title() for col in skill_sums.index]
        
        fig_skills = px.bar(
            x=skill_sums.values,
            y=skill_names,
            orientation='h',
            title="Most In-Demand Skills",
            labels={'x': 'Frequency', 'y': 'Skill'}
        )
        fig_skills.update_layout(height=500)
        st.plotly_chart(fig_skills, use_container_width=True)

def create_filters(df):
    """Create sidebar filters."""
    st.sidebar.header("🔍 Filters")
    
    filters = {}
    
    # Title filter
    if 'title' in df.columns:
        unique_titles = sorted(df['title'].dropna().unique())
        if unique_titles:
            selected_titles = st.sidebar.multiselect(
                "Job Titles",
                options=unique_titles,
                default=[]
            )
            if selected_titles:
                filters['title'] = selected_titles
    
    # Company filter
    if 'company' in df.columns:
        unique_companies = sorted(df['company'].dropna().unique())
        if unique_companies:
            selected_companies = st.sidebar.multiselect(
                "Companies",
                options=unique_companies,
                default=[]
            )
            if selected_companies:
                filters['company'] = selected_companies
    
    # Location filter
    if 'location' in df.columns:
        unique_locations = sorted(df['location'].dropna().unique())
        if unique_locations:
            selected_locations = st.sidebar.multiselect(
                "Locations",
                options=unique_locations,
                default=[]
            )
            if selected_locations:
                filters['location'] = selected_locations
    
    # Cluster filter
    if 'cluster' in df.columns:
        unique_clusters = sorted(df['cluster'].dropna().unique())
        if unique_clusters:
            selected_clusters = st.sidebar.multiselect(
                "Job Categories",
                options=[f"Cluster {c}" for c in unique_clusters],
                default=[]
            )
            if selected_clusters:
                # Extract cluster numbers
                cluster_nums = [int(c.split()[1]) for c in selected_clusters]
                filters['cluster'] = cluster_nums
    
    # Skills filter
    skill_columns = [col for col in df.columns if col.startswith('skill_')]
    if skill_columns:
        st.sidebar.subheader("Skills")
        selected_skills = []
        
        # Show top skills as checkboxes
        skill_sums = df[skill_columns].sum().sort_values(ascending=False).head(10)
        for skill_col in skill_sums.index:
            skill_name = skill_col.replace('skill_', '').replace('_', ' ').title()
            if st.sidebar.checkbox(skill_name):
                selected_skills.append(skill_col)
        
        if selected_skills:
            filters['skills'] = selected_skills
    
    # Date filter
    if 'date_posted' in df.columns:
        st.sidebar.subheader("Date Range")
        date_col = df['date_posted'].dropna()
        if not date_col.empty:
            min_date = date_col.min().date()
            max_date = date_col.max().date()
            
            selected_date_range = st.sidebar.date_input(
                "Posted Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(selected_date_range) == 2:
                filters['date_range'] = selected_date_range
    
    return filters

def apply_filters(df, filters):
    """Apply selected filters to DataFrame."""
    filtered_df = df.copy()
    
    for filter_key, filter_value in filters.items():
        if filter_key == 'title' and filter_value:
            filtered_df = filtered_df[filtered_df['title'].isin(filter_value)]
        elif filter_key == 'company' and filter_value:
            filtered_df = filtered_df[filtered_df['company'].isin(filter_value)]
        elif filter_key == 'location' and filter_value:
            filtered_df = filtered_df[filtered_df['location'].isin(filter_value)]
        elif filter_key == 'cluster' and filter_value:
            filtered_df = filtered_df[filtered_df['cluster'].isin(filter_value)]
        elif filter_key == 'skills' and filter_value:
            # Filter for jobs that have at least one of the selected skills
            skill_mask = filtered_df[filter_value].sum(axis=1) > 0
            filtered_df = filtered_df[skill_mask]
        elif filter_key == 'date_range' and filter_value:
            start_date, end_date = filter_value
            date_mask = (
                (filtered_df['date_posted'].dt.date >= start_date) &
                (filtered_df['date_posted'].dt.date <= end_date)
            )
            filtered_df = filtered_df[date_mask]
    
    return filtered_df

def display_job_table(df):
    """Display job data table with search and pagination."""
    st.subheader("📋 Job Listings")
    
    # Search functionality
    search_term = st.text_input("🔍 Search jobs (title, company, description):", "")
    
    if search_term:
        search_mask = (
            df['title'].str.contains(search_term, case=False, na=False) |
            df['company'].str.contains(search_term, case=False, na=False) |
            df.get('description', pd.Series()).str.contains(search_term, case=False, na=False)
        )
        df = df[search_mask]
    
    # Select columns to display
    display_columns = ['title', 'company', 'location']
    
    # Add additional columns if they exist
    optional_columns = ['salary', 'date_posted', 'cluster', 'cluster_label']
    for col in optional_columns:
        if col in df.columns:
            display_columns.append(col)
    
    # Filter out columns that don't exist
    display_columns = [col for col in display_columns if col in df.columns]
    
    if df.empty:
        st.info("No jobs match the current filters.")
        return
    
    # Display summary
    st.write(f"Showing {len(df)} jobs")
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by:",
        options=display_columns,
        index=0
    )
    
    sort_ascending = st.checkbox("Ascending", value=True)
    
    # Sort dataframe
    df_sorted = df.sort_values(by=sort_by, ascending=sort_ascending)
    
    # Display table
    st.dataframe(
        df_sorted[display_columns],
        use_container_width=True,
        height=400
    )
    
    # Export functionality
    if st.button("📥 Export Filtered Data to CSV"):
        csv = df_sorted.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"filtered_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def display_job_details(df):
    """Display detailed view for selected job."""
    st.subheader("🔍 Job Details")
    
    if df.empty:
        st.info("No jobs to display")
        return
    
    # Job selection
    job_options = [
        f"{row['title']} at {row['company']} ({row.name})"
        for _, row in df.iterrows()
    ]
    
    if not job_options:
        st.info("No jobs available")
        return
    
    selected_job = st.selectbox(
        "Select a job to view details:",
        options=job_options,
        index=0
    )
    
    # Extract job index
    job_index = int(selected_job.split('(')[-1].split(')')[0])
    job_row = df.loc[job_index]
    
    # Display job details
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Title:**", job_row.get('title', 'N/A'))
        st.write("**Company:**", job_row.get('company', 'N/A'))
        st.write("**Location:**", job_row.get('location', 'N/A'))
        
        if 'salary' in job_row and pd.notna(job_row['salary']):
            st.write("**Salary:**", job_row['salary'])
        
        if 'date_posted' in job_row and pd.notna(job_row['date_posted']):
            st.write("**Posted:**", job_row['date_posted'].strftime('%Y-%m-%d'))
    
    with col2:
        if 'cluster' in job_row and pd.notna(job_row['cluster']):
            st.write("**Category:**", f"Cluster {job_row['cluster']}")
        
        if 'cluster_label' in job_row and pd.notna(job_row['cluster_label']):
            st.write("**Category Label:**", job_row['cluster_label'])
        
        if 'url' in job_row and pd.notna(job_row['url']):
            st.write("**Job URL:**", job_row['url'])
    
    # Description
    if 'description' in job_row and pd.notna(job_row['description']):
        st.write("**Description:**")
        st.text_area("", job_row['description'], height=200, disabled=True)
    
    # Skills
    skill_columns = [col for col in df.columns if col.startswith('skill_')]
    if skill_columns:
        job_skills = []
        for skill_col in skill_columns:
            if job_row[skill_col] > 0:  # Assuming binary or frequency values
                skill_name = skill_col.replace('skill_', '').replace('_', ' ').title()
                job_skills.append(skill_name)
        
        if job_skills:
            st.write("**Required Skills:**")
            st.write(", ".join(job_skills))

def main():
    """Main Streamlit app."""
    # Page configuration
    st.set_page_config(
        page_title="Job Monitoring Dashboard",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("💼 Job Monitoring Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading job data..."):
        df = load_data()
    
    if df is None:
        st.error("❌ No job data found. Please run the pipeline first to scrape and process job data.")
        st.info("Run: `python src/main_pipeline.py initial` to get started")
        return
    
    if df.empty:
        st.warning("⚠️ Job data file is empty.")
        return
    
    # Show last update time
    data_path = "data/processed_jobs.csv"
    if os.path.exists(data_path):
        mod_time = datetime.fromtimestamp(os.path.getmtime(data_path))
        st.info(f"📅 Data last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create filters
    filters = create_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Display metrics
    create_metrics_row(filtered_df)
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Job Table", "🔍 Job Details", "📈 Analytics"])
    
    with tab1:
        if not filtered_df.empty:
            create_visualizations(filtered_df)
        else:
            st.info("No data to display with current filters.")
    
    with tab2:
        display_job_table(filtered_df)
    
    with tab3:
        display_job_details(filtered_df)
    
    with tab4:
        st.subheader("📈 Advanced Analytics")
        
        if not filtered_df.empty:
            # Time series analysis if date column exists
            if 'date_posted' in filtered_df.columns:
                st.subheader("Jobs Posted Over Time")
                
                # Group by date
                date_counts = filtered_df.groupby(filtered_df['date_posted'].dt.date).size()
                
                if not date_counts.empty:
                    fig_timeline = px.line(
                        x=date_counts.index,
                        y=date_counts.values,
                        title="Job Postings Timeline",
                        labels={'x': 'Date', 'y': 'Number of Jobs'}
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Correlation analysis for skills
            skill_columns = [col for col in filtered_df.columns if col.startswith('skill_')]
            if len(skill_columns) > 1:
                st.subheader("Skill Correlations")
                
                skill_corr = filtered_df[skill_columns].corr()
                
                fig_corr = px.imshow(
                    skill_corr,
                    title="Skill Co-occurrence Matrix",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No data available for analytics with current filters.")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** Use the sidebar filters to narrow down job listings")

if __name__ == "__main__":
    main()
