# Save this code as a Python file (e.g., titanic_dashboard.py)
# Run it from your terminal using: streamlit run titanic_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import numpy as np

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Titanic Data Explorer Dashboard", page_icon="🚢")
st.title("🚢 Enhanced Titanic Data Explorer Dashboard")
st.markdown("""
This interactive dashboard allows exploration of the Titanic dataset using various visualization libraries.
Use the filters in the sidebar to narrow down the data and observe patterns across different plots.
""")

# Custom CSS to improve appearance
st.markdown("""
<style>
div.stPlotlyChart {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border-radius: 5px;
    padding: 1px;
    background-color: #FAFAFA;
}
div.stDataFrame {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border-radius: 5px;
    padding: 1px;
}
div[data-testid="stMetric"] {
    background-color: #f0f8ff;
    border-radius: 7px;
    padding: 15px 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
div[data-testid="stHeader"] {
    background-color: #f6f6f6;
    padding: 10px;
    border-radius: 5px;
    margin-top: 20px;
    margin-bottom: 20px;
}
div.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
div.stTabs [data-baseweb="tab"] {
    background-color: #f0f0f0;
    border-radius: 4px 4px 0px 0px;
    padding: 10px 16px;
    font-weight: 500;
}
div.stTabs [aria-selected="true"] {
    background-color: #e6f3ff;
    border-bottom: 2px solid #4a90e2;
}
</style>
""", unsafe_allow_html=True)

# --- File Uploader ---
uploaded_file = st.file_uploader("📂 Upload your Titanic CSV file", type=["csv"])

# Function to load and prepare data (cached)
@st.cache_data # Cache the output of this function
def load_and_prep_data(file):
    df = pd.read_csv(file)
    df_processed = df.copy() # Work on a copy

    # --- Initial Data Preparation ---
    # (Fill Age, Fare, Embarked; Create FamilySize; Map Sex/Survived etc.)
    if 'Age' in df_processed.columns:
        median_age = df_processed['Age'].median()
        df_processed['Age'].fillna(median_age, inplace=True)
        df_processed['Age'] = pd.to_numeric(df_processed['Age'], errors='coerce') # Ensure numeric
    if 'Fare' in df_processed.columns:
        median_fare = df_processed['Fare'].median()
        df_processed['Fare'].fillna(median_fare, inplace=True)
        df_processed['Fare'] = pd.to_numeric(df_processed['Fare'], errors='coerce') # Ensure numeric
    if 'Embarked' in df_processed.columns:
        # Fill with mode only if missing values exist
        if df_processed['Embarked'].isnull().any():
            mode_embarked = df_processed['Embarked'].mode()[0]
            df_processed['Embarked'].fillna(mode_embarked, inplace=True)
    if 'SibSp' in df_processed.columns and 'Parch' in df_processed.columns:
        df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        # Create a family size category for better visualization
        df_processed['FamilyCategory'] = pd.cut(
            df_processed['FamilySize'],
            bins=[0, 1, 4, 20],
            labels=['Solo', 'Small Family', 'Large Family'],
            right=True # Include the right edge of the bin
        )
    else:
        df_processed['FamilySize'] = 1 # Default if columns missing
        df_processed['FamilyCategory'] = 'Solo'


    if 'Sex' in df_processed.columns:
        # Create numeric version for plots needing it, keep original for filtering/display
        df_processed['Sex_numeric'] = df_processed['Sex'].map({'male': 0, 'female': 1})
    if 'Survived' in df_processed.columns:
        df_processed['Survived'] = pd.to_numeric(df_processed['Survived'], errors='coerce')
        df_processed.dropna(subset=['Survived'], inplace=True) # Crucial: remove rows where survival is unknown
        df_processed['Survived'] = df_processed['Survived'].astype(int)
        # Create categorical status for legends
        df_processed['Survival_Status'] = df_processed['Survived'].map({0: 'Perished', 1: 'Survived'})
    else:
        # Handle case where 'Survived' column is missing entirely
        st.error("The crucial 'Survived' column is missing from the uploaded file. Cannot proceed with survival analysis.")
        return None # Return None to indicate failure

    # Create age groups for better visualization
    if 'Age' in df_processed.columns:
        df_processed['AgeGroup'] = pd.cut(
            df_processed['Age'],
            bins=[0, 12, 18, 35, 60, 100],
            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'],
            right=True # Include the right edge of the bin
        )
    else:
        df_processed['AgeGroup'] = 'Unknown'


    # Map Embarked to readable port names
    if 'Embarked' in df_processed.columns:
        df_processed['EmbarkedLocation'] = df_processed['Embarked'].map({
            'C': 'Cherbourg',
            'Q': 'Queenstown',
            'S': 'Southampton'
        }).fillna('Unknown') # Handle potentially missing Embarked after mode imputation
    else:
        df_processed['EmbarkedLocation'] = 'Unknown'


    # Ensure Pclass is handled, assuming it's always present and numeric/int-like
    if 'Pclass' in df_processed.columns:
        df_processed['Pclass'] = pd.to_numeric(df_processed['Pclass'], errors='coerce')
        df_processed.dropna(subset=['Pclass'], inplace=True)
        df_processed['Pclass'] = df_processed['Pclass'].astype(int)


    # Drop rows with NaN in critical plotting columns after imputation attempt (if any remain)
    critical_cols = ['Age', 'Fare', 'Pclass', 'Sex', 'FamilySize', 'Survived', 'EmbarkedLocation', 'Survival_Status', 'AgeGroup', 'FamilyCategory']
    # Check which critical columns are actually in the dataframe before dropping
    cols_to_check_for_dropna = [col for col in critical_cols if col in df_processed.columns]
    df_processed.dropna(subset=cols_to_check_for_dropna, inplace=True)


    return df_processed

# --- Main App Logic ---
if uploaded_file is not None:
    df_original = load_and_prep_data(uploaded_file)

    # Check if data loading failed (e.g., missing 'Survived' column)
    if df_original is None or df_original.empty:
        if uploaded_file is not None: # Only show this if a file was actually uploaded
            st.error("Could not load or process data. Please check file format and column names.")
        st.stop() # Stop execution if data loading failed critically or resulted in empty df

    st.success("Data loaded and preprocessed successfully!")

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Data")

    # Filter by Pclass (ensure options are from the *original* data)
    # Convert to string for consistent multiselect if Pclass is int
    if 'Pclass' in df_original.columns:
        pclass_options_str = sorted(df_original['Pclass'].astype(str).unique())
        selected_pclass_str = st.sidebar.multiselect(
            'Passenger Class',
            options=pclass_options_str,
            default=pclass_options_str
        )
        # Convert selected Pclass back to int for filtering if they were originally int
        try:
            selected_pclass = [int(p) for p in selected_pclass_str]
        except ValueError:
            st.sidebar.error("Invalid Pclass selected.")
            selected_pclass = [] # Ensure it's empty on error
    else:
        selected_pclass = [] # No Pclass column means no filtering possible


    # Filter by Sex
    if 'Sex' in df_original.columns:
        sex_options = df_original['Sex'].unique()
        selected_sex = st.sidebar.multiselect(
            'Sex',
            options=sex_options,
            default=sex_options
        )
    else:
        selected_sex = []


    # Filter by Embarked
    embarked_col_filter = None # Use a variable to track the actual column used for filtering
    if 'EmbarkedLocation' in df_original.columns and df_original['EmbarkedLocation'].nunique() > 1:
        embarked_options = sorted(df_original['EmbarkedLocation'].dropna().unique()) # Ensure no NaNs in options
        embarked_col_filter = 'EmbarkedLocation'
    elif 'Embarked' in df_original.columns and df_original['Embarked'].nunique() > 1:
        embarked_options = sorted(df_original['Embarked'].dropna().unique()) # Ensure no NaNs in options
        embarked_col_filter = 'Embarked'
    else:
        embarked_options = [] # No embarkation column or only one value


    if embarked_col_filter:
        selected_embarked = st.sidebar.multiselect(
            'Port of Embarkation',
            options=embarked_options,
            default=embarked_options
        )
    else:
        selected_embarked = [] # Cannot filter if no Embarked data

    # Filter by Age Range (Slider) - Use floor/ceil for cleaner range
    if 'Age' in df_original.columns and not df_original['Age'].empty:
        min_age, max_age = int(np.floor(df_original['Age'].min())), int(np.ceil(df_original['Age'].max()))
        selected_age = st.sidebar.slider(
            'Age Range',
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age) # Default range covers all
        )
    else:
        selected_age = (0, 100) # Fallback if no Age column or empty


    # --- Apply Filters to the DataFrame ---
    # Start with the full dataframe
    df_filtered = df_original.copy()

    # Apply filters only if the column exists and filter options are selected
    if 'Pclass' in df_filtered.columns and selected_pclass:
        df_filtered = df_filtered[df_filtered['Pclass'].isin(selected_pclass)]
    if 'Sex' in df_filtered.columns and selected_sex:
        df_filtered = df_filtered[df_filtered['Sex'].isin(selected_sex)]
    if embarked_col_filter and selected_embarked:
        df_filtered = df_filtered[df_filtered[embarked_col_filter].isin(selected_embarked)]
    if 'Age' in df_filtered.columns and selected_age:
        df_filtered = df_filtered[(df_filtered['Age'] >= selected_age[0]) & (df_filtered['Age'] <= selected_age[1])]

    # Check if the filtered dataframe is empty
    if df_filtered.empty:
        st.warning("No data matches the current filter criteria.")
        st.stop() # Stop if no data to plot

    # --- Display Summary Metrics ---
    st.header("📊 Summary Statistics (Filtered Data)")
    if not df_filtered.empty and 'Survived' in df_filtered.columns:
        total_passengers = len(df_filtered)
        survived_count = df_filtered['Survived'].sum() # Assumes Survived is 1 for survived, 0 otherwise
        survival_rate = (survived_count / total_passengers) * 100 if total_passengers > 0 else 0

        # Create more summary metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

        with metrics_col1:
            st.metric(label="Total Passengers", value=f"{total_passengers:,}")
        with metrics_col2:
            st.metric(label="Survival Rate", value=f"{survival_rate:.1f}%")
        # Safely calculate and display Avg Fare
        if 'Fare' in df_filtered.columns and not df_filtered['Fare'].empty:
            with metrics_col3:
                st.metric(label="Avg Fare", value=f"£{df_filtered['Fare'].mean():.1f}")
        else:
            with metrics_col3:
                st.metric(label="Avg Fare", value="N/A")

        # Safely calculate and display Avg Age
        if 'Age' in df_filtered.columns and not df_filtered['Age'].empty:
            with metrics_col4:
                st.metric(label="Avg Age", value=f"{df_filtered['Age'].mean():.1f} yrs")
        else:
            with metrics_col4:
                st.metric(label="Avg Age", value="N/A")


        # Add a mini chart for survival rate by class
        st.subheader("Survival Rate by Class")
        if 'Pclass' in df_filtered.columns and 'Survived' in df_filtered.columns:
            survival_by_class = df_filtered.groupby('Pclass')['Survived'].mean() * 100

            fig_survival_class = px.bar(
                survival_by_class.reset_index(),
                x='Pclass',
                y='Survived',
                labels={'Survived': 'Survival Rate (%)', 'Pclass': 'Passenger Class'},
                text_auto='.1f',
                color='Pclass',
                color_discrete_map={
                    1: '#115f9a',
                    2: '#1984c5',
                    3: '#22a7f0'
                },
                category_orders={"Pclass": sorted(survival_by_class.index.unique())} # Ensure correct order
            )

            fig_survival_class.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                uniformtext_minsize=10,
                uniformtext_mode='hide',
                xaxis_title="Passenger Class",
                yaxis_title="Survival Rate (%)",
                yaxis_range=[0, 100],
                xaxis_type='category',
                template="plotly_white"
            )

            fig_survival_class.update_traces(
                texttemplate='%{text}%',
                textposition='outside'
            )

            st.plotly_chart(fig_survival_class, use_container_width=True)
        else:
            st.info("Cannot display Survival Rate by Class - 'Pclass' or 'Survived' column missing.")


    else:
        st.warning("No data matches the current filter criteria or 'Survived' column is missing.")
        st.stop() # Stop if no data to plot

    # --- Tabs for Different Visualizations ---
    st.header("🎨 Visualizations (Based on Filtered Data)")
    # Updated Tab 1 title
    tab1, tab2, tab3, tab4 = st.tabs(["Numerical Relationships", "Survival Analysis", "Age & Fare Distribution", "Passenger Profiles"])

    # --- Tab 1: Seaborn Pair Plot ---
    with tab1:
        st.subheader("Relationships Between Numerical Features by Survival Status")
        st.markdown("Explore pairwise relationships between Age, Fare, and Family Size, colored by survival status. Diagonal plots show feature distributions.")

        numerical_cols_for_pairplot = ['Age', 'Fare', 'FamilySize']
        required_cols_pairplot = numerical_cols_for_pairplot + ['Survival_Status']

        # Check if required columns are in the filtered data
        if all(col in df_filtered.columns for col in required_cols_pairplot) and not df_filtered.empty:
            try:
                # Create a matplotlib figure explicitly
                # Use a smaller size if corner=False, larger if corner=True or adding 'col'
                # Adjust height based on number of vars, aspect for shape
                g = sns.pairplot(
                    df_filtered,
                    vars=numerical_cols_for_pairplot,
                    hue='Survival_Status',
                    palette={'Perished': '#c23728', 'Survived': '#2e8b57'},
                    diag_kind='kde', # Use KDE plots on the diagonal
                    corner=True, # Show only the lower triangle
                    height=3, # Height of each facet
                    aspect=1 # Aspect ratio of each facet
                )

                # Improve titles and labels if needed (pairplot handles internal titles well)
                g.fig.suptitle('Pairwise Relationships by Survival Status', y=1.02, fontsize=16) # Add a main title
                # fig_pairplot.fig.set_size_inches(10, 10) # pairplot returns a FacetGrid, adjust via height/aspect instead

                # Display the plot in Streamlit
                st.pyplot(g.fig) # Get the figure from the FacetGrid
                plt.close(g.fig) # Close the figure to free memory
            except Exception as e:
                st.error(f"Error generating Pair Plot: {e}")
        else:
            missing = [col for col in required_cols_pairplot if col not in df_filtered.columns]
            if missing:
                st.warning(f"Cannot generate Pair Plot. Missing columns: {', '.join(missing)}.")
            else:
                st.warning("Cannot generate Pair Plot - Filtered data is empty.")


    # --- Tab 2: Enhanced Survival Analysis Plots ---
    with tab2:
        st.subheader("Survival Analysis by Key Factors")

        col1, col2 = st.columns(2)

        with col1:
            # Survival by sex and class heatmap
            required_cols_heatmap = ['Sex', 'Pclass', 'Survived']
            if all(col in df_filtered.columns for col in required_cols_heatmap) and not df_filtered.empty:
                try:
                    # Calculate survival rate by sex and class
                    # Ensure Pclass is treated as category/object before pivot_table if needed, or ensure pivot_table handles integers
                    survival_by_sex_class = df_filtered.pivot_table(
                        index='Sex',
                        columns='Pclass',
                        values='Survived',
                        aggfunc='mean'
                    ) * 100

                    fig_heatmap = px.imshow(
                        survival_by_sex_class,
                        text_auto='.1f',
                        labels=dict(x="Passenger Class", y="Sex", color="Survival Rate (%)"),
                        color_continuous_scale="RdYlGn",
                        zmin=0,
                        zmax=100,
                        aspect="auto",
                        # Ensure column order matches Pclass 1, 2, 3
                        x=[str(c) for c in sorted(df_filtered['Pclass'].unique())] # Use string representation for labels if Pclass is int
                    )

                    fig_heatmap.update_traces(
                        texttemplate="%{text}%",
                        textfont={"size": 14}
                    )

                    fig_heatmap.update_layout(
                        title="Survival Rate (%) by Sex and Class",
                        xaxis=dict(title="Passenger Class"), # Plotly handles categorical x-axis automatically from data
                        coloraxis_colorbar=dict(title="Survival<br>Rate (%)")
                    )

                    st.plotly_chart(fig_heatmap, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating heatmap: {e}")
            else:
                missing = [col for col in required_cols_heatmap if col not in df_filtered.columns]
                if missing:
                    st.warning(f"Cannot generate heatmap. Missing columns: {', '.join(missing)}.")
                else:
                    st.warning("Cannot generate heatmap - Filtered data is empty.")


        with col2:
            # Age group survival chart
            required_cols_age_survival = ['AgeGroup', 'Survived']
            if all(col in df_filtered.columns for col in required_cols_age_survival) and not df_filtered.empty:
                try:
                    # Ensure AgeGroup order
                    age_group_order = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
                    # Calculate survival rate, keeping all age groups even if count is 0 after filtering
                    age_survival_counts = df_filtered.groupby('AgeGroup').size().reindex(age_group_order, fill_value=0).reset_index(name='Count')
                    age_survival_rates = df_filtered.groupby('AgeGroup')['Survived'].mean().reindex(age_group_order, fill_value=0).reset_index(name='Survived')
                    age_survival = pd.merge(age_survival_rates, age_survival_counts, on='AgeGroup')
                    age_survival['SurvivalPercentage'] = age_survival['Survived'] * 100


                    fig_age_survival = px.bar(
                        age_survival,
                        x='AgeGroup',
                        y='SurvivalPercentage',
                        text_auto='.1f',
                        color='SurvivalPercentage',
                        color_continuous_scale="RdYlGn",
                        labels={'SurvivalPercentage': 'Survival Rate (%)', 'AgeGroup': 'Age Group'},
                        hover_data=['Count'],
                        category_orders={"AgeGroup": age_group_order} # Explicitly set order
                    )

                    fig_age_survival.update_traces(
                        texttemplate='%{text}%',
                        textposition='outside'
                    )

                    fig_age_survival.update_layout(
                        title="Survival Rate by Age Group",
                        xaxis_title="Age Group",
                        yaxis_title="Survival Rate (%)",
                        yaxis_range=[0, 100],
                        coloraxis_showscale=False
                    )

                    st.plotly_chart(fig_age_survival, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating age group chart: {e}")
            else:
                missing = [col for col in required_cols_age_survival if col not in df_filtered.columns]
                if missing:
                    st.warning(f"Cannot generate age group chart. Missing columns: {', '.join(missing)}.")
                else:
                    st.warning("Cannot generate age group chart - Filtered data is empty.")

        # Family size effect on survival
        required_cols_family_survival = ['Pclass', 'FamilyCategory', 'Survived']
        if all(col in df_filtered.columns for col in required_cols_family_survival) and not df_filtered.empty:
            try:
                # Ensure FamilyCategory and Pclass order
                family_category_order = ['Solo', 'Small Family', 'Large Family']
                pclass_order = sorted(df_filtered['Pclass'].unique()) # Use unique classes from filtered data

                family_survival = df_filtered.groupby(['Pclass', 'FamilyCategory'])['Survived'].mean().reset_index()
                family_survival['SurvivalPercentage'] = family_survival['Survived'] * 100

                # Add missing combinations with 0 survival rate if needed for complete bars
                # This part can be complex if a category/class combination is entirely filtered out.
                # For simplicity, the plot will only show combinations present in filtered data.

                fig_family = px.bar(
                    family_survival,
                    x='FamilyCategory',
                    y='SurvivalPercentage',
                    color='Pclass',
                    barmode='group',
                    text_auto='.1f',
                    labels={
                        'SurvivalPercentage': 'Survival Rate (%)',
                        'FamilyCategory': 'Family Size',
                        'Pclass': 'Passenger Class'
                    },
                    color_discrete_map={
                        1: '#115f9a',
                        2: '#1984c5',
                        3: '#22a7f0'
                    },
                    category_orders={
                        "FamilyCategory": family_category_order,
                        "Pclass": pclass_order
                    }
                )

                fig_family.update_traces(
                    texttemplate='%{text}%',
                    textposition='outside'
                )

                fig_family.update_layout(
                    title="Survival Rate by Family Size and Class",
                    xaxis_title="Family Size Category",
                    yaxis_title="Survival Rate (%)",
                    yaxis_range=[0, 100],
                    legend_title="Passenger Class"
                )

                st.plotly_chart(fig_family, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating family size chart: {e}")
        else:
            missing = [col for col in required_cols_family_survival if col not in df_filtered.columns]
            if missing:
                st.warning(f"Cannot generate family size chart. Missing columns: {', '.join(missing)}.")
            else:
                st.warning("Cannot generate family size chart - Filtered data is empty.")


    # --- Tab 3: Enhanced Seaborn Violin Plot ---
    with tab3:
        st.subheader("Age & Fare Distribution Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Enhanced violin plot for age distribution
            required_cols_seaborn_age = ['Age', 'Sex', 'Pclass', 'Survival_Status', 'Survived']
            if all(col in df_filtered.columns for col in required_cols_seaborn_age) and not df_filtered.empty:
                try:
                    # Create a matplotlib figure explicitly for st.pyplot
                    fig_seaborn, ax = plt.subplots(figsize=(10, 6))

                    # Set more appealing visual style
                    sns.set_style("whitegrid")
                    # Use a consistent palette, maybe match Plotly colors or define a new one
                    # sns.set_palette("deep") # Or use a custom palette

                    sns.violinplot(
                        data=df_filtered,
                        x='Pclass',
                        y='Age',
                        hue='Survival_Status',
                        order=sorted(df_filtered['Pclass'].unique()),
                        split=True,
                        palette={'Perished': '#c23728', 'Survived': '#2e8b57'},
                        inner='quartile',
                        linewidth=1.5,
                        ax=ax
                    )

                    ax.set_title('Age Distribution by Class and Survival', fontsize=16, pad=20)
                    ax.set_xlabel('Passenger Class', fontsize=14)
                    ax.set_ylabel('Age', fontsize=14)
                    ax.legend(title='Outcome', title_fontsize=12, fontsize=11)

                    # Add some styling
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.8)

                    plt.tight_layout()
                    st.pyplot(fig_seaborn)
                    plt.close(fig_seaborn) # Close the figure
                except Exception as e:
                    st.error(f"Error generating Seaborn plot: {e}")
            else:
                missing = [col for col in required_cols_seaborn_age if col not in df_filtered.columns]
                if missing:
                    st.warning(f"Cannot generate Age Distribution plot. Missing columns: {', '.join(missing)}.")
                else:
                    st.warning("Cannot generate Age Distribution plot - Filtered data is empty.")


        with col2:
            # Fare distribution by class
            required_cols_seaborn_fare = ['Fare', 'Pclass', 'Survival_Status']
            if all(col in df_filtered.columns for col in required_cols_seaborn_fare) and not df_filtered.empty:
                try:
                    fig_fare, ax = plt.subplots(figsize=(10, 6))

                    sns.boxplot(
                        data=df_filtered,
                        x='Pclass',
                        y='Fare',
                        hue='Survival_Status',
                        palette={'Perished': '#c23728', 'Survived': '#2e8b57'},
                        fliersize=3,
                        linewidth=1.5,
                        ax=ax
                    )

                    ax.set_title('Fare Distribution by Class and Survival', fontsize=16, pad=20)
                    ax.set_xlabel('Passenger Class', fontsize=14)
                    ax.set_ylabel('Fare (£)', fontsize=14)
                    ax.legend(title='Outcome', title_fontsize=12, fontsize=11)

                    # Log scale for better visualization of fare distribution
                    ax.set_yscale('log')

                    # Add some styling
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.8)

                    plt.tight_layout()
                    st.pyplot(fig_fare)
                    plt.close(fig_fare) # Close the figure
                except Exception as e:
                    st.error(f"Error generating fare boxplot: {e}")
            else:
                missing = [col for col in required_cols_seaborn_fare if col not in df_filtered.columns]
                if missing:
                    st.warning(f"Cannot generate fare distribution plot. Missing required columns: {', '.join(missing)}.")
                else:
                    st.warning("Cannot generate fare distribution plot - Filtered data is empty.")

    # --- Tab 4: Enhanced Altair Scatter Plot ---
    with tab4:
        st.subheader("Interactive Passenger Profile Explorer")
        st.markdown("Explore relationships between Age, Fare, and Class, colored by survival status. *Hover for details, zoom/pan.*")

        # Check required columns
        required_cols_altair = ['Age', 'Fare', 'Survival_Status', 'Sex', 'Pclass']
        # Add 'Survived' as it's used for tooltips implicitly often
        required_cols_altair_tooltip_check = required_cols_altair + ['Survived']


        # Add columns for tooltips if they exist
        tooltip_cols = [col for col in ['Age', 'Fare', 'Survival_Status', 'Sex', 'Pclass'] if col in df_filtered.columns]
        if 'Name' in df_filtered.columns:
            tooltip_cols.append('Name')
        if 'EmbarkedLocation' in df_filtered.columns:
            tooltip_cols.append('EmbarkedLocation')
        if 'FamilySize' in df_filtered.columns:
            tooltip_cols.append('FamilySize')


        # Let user choose log scale option for fare
        use_log_scale = st.checkbox("Use logarithmic scale for Fare", value=True, key='altair_log_scale')

        if all(col in df_filtered.columns for col in required_cols_altair_tooltip_check) and not df_filtered.empty:
            try:
                # Create a size variable based on family size for additional dimension
                alt_df = df_filtered.copy()
                if 'FamilySize' not in alt_df.columns:
                    alt_df['FamilySize'] = 1 # Default if column is missing

                # Determine the Y-axis scale type based on checkbox
                y_scale = alt.Scale(type='log', domainMin=1) if use_log_scale else alt.Scale(zero=False) # domainMin prevents issues with log(0)

                chart_altair = alt.Chart(alt_df).mark_circle(opacity=0.8).encode(
                    x=alt.X('Age', scale=alt.Scale(zero=False), title='Age (years)'),
                    y=alt.Y('Fare', scale=y_scale, title='Fare (£)'),
                    size=alt.Size('FamilySize', scale=alt.Scale(range=[30, 300]), title='Family Size'),
                    color=alt.Color(
                        'Survival_Status',
                        scale=alt.Scale(domain=['Perished', 'Survived'], range=['#c23728', '#2e8b57']),
                        legend=alt.Legend(title="Survival Status")
                    ),
                    shape=alt.Shape('Sex', scale=alt.Scale(domain=['male', 'female'], range=['triangle-up', 'circle']), title='Sex'), # Use appropriate shapes
                    tooltip=tooltip_cols # Use the dynamically created tooltip columns
                ).properties(
                    title='Age vs Fare by Survival Status',
                    height=500
                ).interactive() # Enable interactive features (zoom, pan)

                # Add a layer for better class visualization as text labels near points
                # Only add if Pclass is available
                if 'Pclass' in alt_df.columns:
                    text_layer = alt.Chart(alt_df).mark_text(
                        align='center',
                        baseline='middle',
                        fontSize=10,
                        fontWeight='bold',
                        dx=0, # Adjust text position relative to circle if needed
                        dy=10 # Place text slightly below the circle
                    ).encode(
                        x=alt.X('Age'),
                        y=alt.Y('Fare'),
                        text=alt.Text('Pclass:N'), # Use nominal type for discrete values
                        opacity=alt.value(0.7),
                        order='Pclass', # Helps with rendering order
                        tooltip=tooltip_cols # Also include tooltips on text layer
                    )
                    # Combine layers
                    final_chart = (chart_altair + text_layer).configure_axis(
                        labelFontSize=12,
                        titleFontSize=14
                    ).configure_legend(
                        labelFontSize=12,
                        titleFontSize=14
                    ).configure_title(
                        fontSize=16,
                        anchor='middle'
                    )
                else:
                    # If Pclass is not available, just use the scatter plot
                    final_chart = chart_altair.configure_axis(
                        labelFontSize=12,
                        titleFontSize=14
                    ).configure_legend(
                        labelFontSize=12,
                        titleFontSize=14
                    ).configure_title(
                        fontSize=16,
                        anchor='middle'
                    )


                st.altair_chart(final_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating Altair plot: {e}")
        else:
            missing = [col for col in required_cols_altair_tooltip_check if col not in df_filtered.columns]
            if missing:
                st.warning(f"Cannot generate Altair plot. Missing essential columns: {', '.join(missing)}.")
            else:
                st.warning("Cannot generate Altair plot - Filtered data is empty.")


    # --- Display Filtered Data Table (Optional) ---
    with st.expander("View Filtered Data Table", expanded=False):
        if not df_filtered.empty:
            st.dataframe(
                df_filtered.style.background_gradient(
                    subset=['Age', 'Fare'],
                    cmap='Blues'
                ).highlight_max(
                    subset=['Age', 'Fare'],
                    color='lightgreen'
                ).highlight_min(
                    subset=['Age', 'Fare'],
                    color='lightcoral'
                ),
                height=400
            )
        else:
            st.info("Filtered data table is empty.")


else:
    st.info("☝️ Upload a Titanic CSV file to start the exploration!")

    # Add example image of the dashboard
    st.markdown("""
    ### What to expect:
    - Interactive visualizations with Plotly, Seaborn, and Altair
    - Comprehensive survival analysis by various factors
    - Intuitive filtering options
    - Detailed passenger profiles

    *(Example datasets can often be found on Kaggle: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data))*
    """)

    # Add example thumbnail with placeholder
    # Make sure you have this image path or replace with a public URL
    # st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/data/titanic.jpg", caption="Example Titanic Dataset Visualization", use_column_width=True)
    # Using a more reliable placeholder or remove if no image is available
    # st.markdown("*(Image preview of the dashboard goes here)*")