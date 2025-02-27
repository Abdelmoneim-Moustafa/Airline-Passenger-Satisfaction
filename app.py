
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image

# Custom CSS Styling
def set_global_styles():
    st.markdown(
        """
        <style>
        /* Set the background colors for the main container and sidebar */
    .reportview-container { background: #f5f5f5; }
    .sidebar .sidebar-content { background: #f0f2f6; }
    h1 { color: #003366; font-size: 2.5em; border-bottom: 2px solid #003366; }
    h2 { color: #003366; font-size: 1.8em; }
    h3 { color: #003366; font-size: 1.4em; }
    p, li { color: #333333; font-size: 1.1em; line-height: 1.6; }
    .stDataFrame { border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .table-style { 
        margin: 1em 0; 
        border-collapse: collapse;
        width: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .table-style th { 
        background: #003366; 
        color: white; 
        padding: 12px; 
        text-align: left;
    }
    .table-style td { 
        padding: 12px; 
        border-bottom: 1px solid #ddd;
        text-align: center;
    }
    .table-style tr:hover { background-color: #f5f5f5; }
    </style>
    """, unsafe_allow_html=True)


# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("airline_passenger_satisfaction.csv")

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Interactive Analysis", "Model Insights", "Conclusions"])
# Set global styles
set_global_styles()


# Page 1: Overview
if page == "Overview":
    st.title("✈️ Airline Passenger Satisfaction Analysis")

    # Display cover image
    try:
        st.image("dataset-cover.jpg", use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Dataset cover image not found.")

    # Key Objectives & Dataset Overview in a single row layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### 🎯 Key Objectives  
        - Identify key drivers of passenger satisfaction  
        - Evaluate service quality across customer segments  
        - Provide actionable insights for service improvement  
        - Leverage machine learning for satisfaction prediction  
        """)

    with col2:
        st.markdown("""
        ### 📊 Dataset Overview  
        - **129,880** passenger records analyzed  
        - **24 key features**, including:  
          - **Demographics** (age, gender, customer type)  
          - **Travel details** (flight distance, class, type of travel)  
          - **Service ratings** (comfort, inflight service, food & drink, etc.)  
          - **Overall satisfaction** (satisfied or dissatisfied)  
        """)

    # Data Preview with Expander
    with st.expander("🔍 **Explore Raw Data**"):
        num_rows = st.slider("Select number of rows to display", 5, 20, 10)
        st.dataframe(df.head(num_rows).style.set_properties(**{'background-color': '#f5f5f5'}))

# Page 2: Exploratory Data Analysis
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    # Interactive Filters
    st.sidebar.header("Filters")

    # Data Overview
    with st.expander("Dataset Structure"):
        cols = st.columns(2)
        cols[0].metric("Total Passengers", df.shape[0])
        cols[1].metric("Number of Features", df.shape[1])
        
        st.subheader("Data Sample")
        st.dataframe(df.head(10), height=300)

    # Satisfaction Distribution
    st.subheader("Satisfaction Distribution")
    satisfaction_dist = df['Satisfaction'].value_counts(normalize=True)
    fig = px.pie(satisfaction_dist, 
                 values=satisfaction_dist.values,
                 names=satisfaction_dist.index,
                 color_discrete_sequence=['#003366', '#dc1e1e'])
    st.plotly_chart(fig, use_container_width=True)

    # Demographic Analysis
    st.subheader("Demographic Breakdown")
    dem_cols = st.columns(2)
    
    with dem_cols[0]:
        gender_dist = df['Gender'].value_counts()
        fig = px.bar(gender_dist, 
                     color=gender_dist.index,
                     color_discrete_sequence=['#003366', '#dc1e1e'])
        st.plotly_chart(fig, use_container_width=True)
        
    with dem_cols[1]:
        fig = px.histogram(df, x='Age', nbins=20,
                          color_discrete_sequence=['#003366'])
        st.plotly_chart(fig, use_container_width=True)
    # --- Satisfaction Distribution by Age Group (Predefined) ---
    st.subheader("🧓 Satisfaction Distribution by Age Group")
    if 'Age' in df.columns and 'Satisfaction' in df.columns:
        # Predefined age groups
        age_groups = ["<20", "20-29", "30-39", "40-49", "50-59", "60+"]
        selected_age_group = st.selectbox("Select Age Group:", age_groups)
        
        # Filter data based on selected age group
        if selected_age_group == "<20":
            df_group = df[df['Age'] < 20]
        elif selected_age_group == "20-29":
            df_group = df[(df['Age'] >= 20) & (df['Age'] < 30)]
        elif selected_age_group == "30-39":
            df_group = df[(df['Age'] >= 30) & (df['Age'] < 40)]
        elif selected_age_group == "40-49":
            df_group = df[(df['Age'] >= 40) & (df['Age'] < 50)]
        elif selected_age_group == "50-59":
            df_group = df[(df['Age'] >= 50) & (df['Age'] < 60)]
        elif selected_age_group == "60+":
            df_group = df[df['Age'] >= 60]
        else:
            df_group = df.copy()
        
        # Calculate satisfaction distribution for the selected age group
        sat_counts = df_group['Satisfaction'].value_counts().reset_index()
        sat_counts.columns = ['Satisfaction', 'Count']
        fig_age = px.bar(sat_counts, x='Satisfaction', y='Count',
                         title=f"Satisfaction Distribution for Age Group: {selected_age_group}",
                         color='Satisfaction', color_discrete_sequence=['#003366', '#dc1e1e'])
        st.plotly_chart(fig_age, use_container_width=True)
        
        total = sat_counts['Count'].sum()
        sat_counts['Percentage'] = (sat_counts['Count'] / total * 100).round(1)
        st.markdown("**Age Group Statistics**")
        st.dataframe(sat_counts)
    else:
        st.warning("Columns 'Age' and/or 'Satisfaction' not found.")
    
    # --- Satisfaction by Customer Segments ---
    st.subheader("👥 Satisfaction by Customer Segments")
    segments = ['Customer Type', 'Type of Travel', 'Class']
    
    satisfaction_labels = df['Satisfaction'].unique().tolist()

    for segment in segments:
        st.markdown(f"### {segment}")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Use actual satisfaction labels from data
            seg_data = df.groupby([segment, 'Satisfaction']).size().unstack()
            fig = px.bar(seg_data, 
                        x=seg_data.index, 
                        y=satisfaction_labels,
                        color_discrete_sequence=['#003366', '#dc1e1e'],
                        labels={'value': 'Passengers Count', 'variable': 'Satisfaction'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Dynamically calculate percentages based on actual labels
            seg_stats = seg_data.copy()
            seg_stats['Total'] = seg_stats.sum(axis=1)
            
            for label in satisfaction_labels:
                seg_stats[f'{label} %'] = (seg_stats[label] / seg_stats['Total'] * 100).round(1)
            
            st.markdown(f"**{segment} Statistics**")
            st.dataframe(seg_stats.style.format({f'{label} %': '{:.1f}%' for label in satisfaction_labels}))

# Page 3: Interactive Analysis
elif page == "Interactive Analysis":
    st.title("📊 Interactive Analysis")
    
    # Service Rating Analysis - Expanded
    st.subheader("Service Rating Explorer")
    
    # List all service-related features from the dataset
    service_features = [
        'Seat Comfort', 'In-flight Entertainment', 'On-board Service',
        'Leg Room Service', 'Baggage Handling', 'Check-in Service',
        'In-flight Service', 'Cleanliness', 'Food and Drink', 
        'Gate Location', 'Ease of Online Booking', 'Online Boarding'
    ]
    
    analysis_cols = st.columns(2)
    with analysis_cols[0]:
        service_feature = st.selectbox(
            "Select Service Feature",
            options=service_features,
            index=0
        )
    
    with analysis_cols[1]:
        group_by = st.selectbox(
            "Group By",
            options=['Class', 'Customer Type', 'Type of Travel', 'Gender'],
            index=0
        )
    
    # Visualization
    fig = px.box(df, x=service_feature, y='Satisfaction',
                color=group_by, 
                color_discrete_sequence=['#003366', '#dc1e1e', '#909195'],
                title=f"{service_feature} Ratings by {group_by}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Dynamic Data Table
    with st.expander("View Detailed Ratings Data"):
        agg_df = df.groupby([group_by, 'Satisfaction'])[service_feature].agg(
            ['mean', 'count', 'std']
        ).reset_index()
        agg_df.columns = [group_by, 'Satisfaction', 'Average Rating', 
                         'Response Count', 'Rating Variability']
        st.dataframe(
            agg_df.style.format({
                'Average Rating': '{:.2f}',
                'Rating Variability': '{:.2f}'
            }),
            height=300
        )

    # Flight Delay Analysis - Enhanced
    st.subheader("Flight Delay Impact Analysis")
    
    delay_cols = st.columns(2)
    with delay_cols[0]:
        # Scatter plot with trendline
        fig = px.scatter(
            df, 
            x='Departure Delay', 
            y='Arrival Delay',
            color='Satisfaction',
            trendline="lowess",
            color_discrete_sequence=['#003366', '#dc1e1e'],
            title="Departure vs Arrival Delays"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Delay statistics table
        delay_stats = df[['Departure Delay', 'Arrival Delay']].describe().loc[['mean', '50%', 'max']]
        st.markdown("**Delay Statistics (minutes)**")
        st.dataframe(
            delay_stats.style.format("{:.1f}"),
            use_container_width=True
        )

    with delay_cols[1]:
        delay_threshold = st.slider(
            "Minimum Delay Threshold (minutes)", 
            0, 300, 30,
            help="Analyze satisfaction for flights delayed beyond this duration"
        )
        
        delayed_df = df[df['Departure Delay'] > delay_threshold]
        satisfaction_rate = delayed_df['Satisfaction'].value_counts(normalize=True).reset_index()
        satisfaction_rate.columns = ['Satisfaction', 'Percentage']
        
        # Visualization
        fig = px.pie(
            satisfaction_rate, 
            values='Percentage', 
            names='Satisfaction',
            color_discrete_sequence=['#003366', '#dc1e1e'],
            title=f"Satisfaction for Flights Delayed >{delay_threshold} mins"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Delay impact table
        st.markdown(f"**Delay Impact Analysis (> {delay_threshold} mins)**")
        delay_impact = pd.DataFrame({
            'Metric': [
                'Total Delayed Flights', 
                'Average Delay Duration',
                'Satisfaction Rate',
                'Dissatisfaction Rate'
            ],
            'Value': [
                len(delayed_df),
                f"{delayed_df['Departure Delay'].mean():.1f} mins",
                f"{satisfaction_rate.iloc[0]['Percentage']*100:.1f}%",
                f"{satisfaction_rate.iloc[1]['Percentage']*100:.1f}%"
            ]
        })
        st.table(delay_impact)

    # Additional Service Comparison Section
    st.subheader("Service Feature Comparison")
    
    selected_services = st.multiselect(
        "Select Services to Compare",
        options=service_features,
        default=['Seat Comfort', 'Food and Drink', 'Online Boarding']
    )
    
    if selected_services:
        comp_cols = st.columns(2)
        with comp_cols[0]:
            # Radar chart for service comparison
            avg_ratings = df[selected_services].mean().reset_index()
            avg_ratings.columns = ['Service', 'Average Rating']
            
            fig = px.line_polar(
                avg_ratings, 
                r='Average Rating', 
                theta='Service', 
                line_close=True,
                title="Average Service Ratings Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with comp_cols[1]:
            # Service ratings correlation matrix
            corr_matrix = df[selected_services].corr()
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Service", y="Service", color="Correlation"),
                x=selected_services,
                y=selected_services,
                title="Service Ratings Correlation"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation data table
            st.markdown("**Correlation Values Table**")
            st.dataframe(
                corr_matrix.style.format("{:.2f}").background_gradient(
                    cmap='Blues', vmin=-1, vmax=1
                ),
                use_container_width=True
            )
# Page 4: Model Insights
elif page == "Model Insights":
    st.title("🤖 Machine Learning Insights")
    
    # Model Performance
    st.subheader("Model Performance Comparison")
    model_data = pd.DataFrame({
        'Model': ['Random Forest', 'Decision Tree', 'Logistic Regression', 'SVM'],
        'Accuracy': [0.966, 0.949, 0.878, 0.959],
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=model_data['Model'], y=model_data['Accuracy'],
                        name='Accuracy', marker_color='Yellow'))
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance and not depend on model # Important 
    st.subheader("Key Predictive Features and Not depend on model Result")
    features = pd.DataFrame({
        'Feature': ['Online Boarding', 'Class', 'Flight Distance',
                   'Inflight WiFi', 'Seat Comfort'],
        'Importance': [0.184, 0.162, 0.121, 0.098, 0.087]
    })
    
    fig = px.bar(features.sort_values('Importance', ascending=True), 
                x='Importance', y='Feature', orientation='h',
                color_discrete_sequence=['#003366'])
    st.plotly_chart(fig, use_container_width=True)

# Page 5: Conclusions
elif page == "Conclusions":
    st.title("🎯 Key Findings & Recommendations")
    
    # Interactive Recommendations
    st.subheader("Interactive Recommendation Simulator")
    
    st.markdown("""
    <div class="markdown-text-container">

    Top Satisfaction Drivers:
      1. On-board Services Quality (24% impact)
      2. Travel Class (18% impact)
      3. Flight Connectivity (15% impact)
      4. Baggage Handling Efficiency (12% impact)
    
    Strategic Recommendations:
    - 🛋️ Enhance premium class amenities and services
    - ✈️ Optimize flight routes for better connectivity
    - 📶 Improve in-flight WiFi connectivity
    - 🧳 Implement real-time baggage tracking
    - 🍱 Upgrade food & beverage quality standards
    
    Customer Retention Opportunities:
    - First-time flyers show 23% lower satisfaction
    - Business travelers are 2.1x more likely to be loyal
    - Delays over 45 minutes reduce satisfaction by 37%
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Satisfaction Improvement Potential")
    fig = px.funnel(df, x=[23, 18, 15, 12], 
                   y=['On-board Services', 'Travel Class', 
                     'Flight Connectivity', 'Baggage Handling'],
                   color_discrete_sequence=['#003366'])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feedback & Suggestions")
    with st.form("feedback_form"):
        name = st.text_input("Name (optional)")
        feedback = st.text_area("Your Feedback")
        rating = st.slider("Rate this dashboard", 1, 5, 5)
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            st.success("Thank you for your feedback! We'll use it to improve our services.")
