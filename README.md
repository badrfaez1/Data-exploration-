# 🚢 Titanic Data Explorer Dashboard

An interactive and comprehensive Streamlit dashboard for exploring the famous Titanic dataset. This application provides multiple visualization perspectives using Plotly, Seaborn, Matplotlib, and Altair to uncover insights about passenger survival patterns.

Link :https://gfe0007group10.streamlit.app/


## Demo


https://github.com/user-attachments/assets/36cc4d16-46ba-4483-be4c-1aeb2ad54887


## ✨ Features

### 📊 Interactive Visualizations
- **Plotly Charts**: Interactive heatmaps, bar charts, and survival analysis
- **Seaborn Plots**: Statistical pair plots and violin distributions
- **Altair Scatter Plots**: Interactive passenger profile explorer with zoom/pan
- **Matplotlib Integration**: Enhanced statistical visualizations

### 🎛️ Dynamic Filtering
- **Passenger Class**: Filter by 1st, 2nd, or 3rd class
- **Sex**: Male/Female passenger filtering
- **Port of Embarkation**: Cherbourg, Queenstown, Southampton
- **Age Range**: Interactive slider for age-based filtering

### 📈 Comprehensive Analysis
- **Survival Statistics**: Real-time metrics with survival rates
- **Age Group Analysis**: Child, Teen, Adult, Senior survival patterns
- **Family Size Impact**: Solo travelers vs. families survival rates
- **Fare Analysis**: Economic factors affecting survival

### 🎨 Professional UI/UX
- Custom CSS styling with modern design
- Tabbed interface for organized content
- Responsive layout with metrics cards
- Interactive data table with styling

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas plotly seaborn matplotlib altair numpy
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/titanic-dashboard.git
   cd titanic-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with:

```
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.15.0
seaborn>=0.12.0
matplotlib>=3.7.0
altair>=5.0.0
numpy>=1.24.0
```

## 📁 Project Structure

```
titanic-dashboard/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Sample data directory (optional)
│   └── titanic.csv
└── assets/               # Static assets (optional)
    └── screenshots/
```

## 📊 Dashboard Sections

### 1. Summary Statistics
- Total passengers count
- Overall survival rate
- Average fare and age
- Class-based survival breakdown

### 2. Numerical Relationships
- Seaborn pair plots showing correlations
- Age, Fare, and Family Size interactions
- Survival status color coding

### 3. Survival Analysis
- Interactive heatmap by sex and class
- Age group survival patterns
- Family size impact visualization

### 4. Age & Fare Distribution
- Violin plots for age distribution
- Box plots for fare analysis
- Logarithmic scaling options

### 5. Passenger Profiles
- Interactive Altair scatter plots
- Multi-dimensional analysis (age, fare, family size)
- Hover tooltips with detailed information

## 💾 Data Requirements

The dashboard expects a CSV file with the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `Survived` | Survival status (0/1) | ✅ Yes |
| `Pclass` | Passenger class (1,2,3) | ✅ Yes |
| `Sex` | Gender (male/female) | ✅ Yes |
| `Age` | Age in years | ✅ Yes |
| `Fare` | Ticket fare | ✅ Yes |
| `SibSp` | Siblings/spouses aboard | ✅ Yes |
| `Parch` | Parents/children aboard | ✅ Yes |
| `Embarked` | Port of embarkation (C/Q/S) | Recommended |
| `Name` | Passenger name | Optional |

### Data Preprocessing
The application automatically handles:
- Missing value imputation (median for Age/Fare, mode for Embarked)
- Feature engineering (FamilySize, Age Groups, Family Categories)
- Data type conversions and validation
- Outlier filtering (Fare >= 1)

## 🎯 Usage Examples

### Basic Usage
1. Launch the application
2. Upload your Titanic CSV file
3. Use sidebar filters to explore specific passenger segments
4. Navigate through tabs to see different visualizations

### Advanced Analysis
- Combine multiple filters to identify survival patterns
- Use the interactive Altair chart to explore individual passengers
- Compare survival rates across different demographic groups
- Export filtered data for further analysis

## 🛡️ Error Handling

The dashboard includes robust error handling for:
- Missing required columns
- Empty datasets after filtering
- Invalid data types
- Visualization rendering errors

## 🔧 Customization

### Adding New Visualizations
```python
# Add to the tabs section
with tab_new:
    st.subheader("Your New Visualization")
    # Your plotting code here
```

### Custom Styling
Modify the CSS in the `st.markdown()` section to change:
- Color schemes
- Card layouts
- Plot containers
- Typography

### Additional Filters
Add new filters in the sidebar section:
```python
# Example: Add cabin filter
if 'Cabin' in df_original.columns:
    cabin_options = df_original['Cabin'].dropna().unique()
    selected_cabin = st.sidebar.multiselect('Cabin', cabin_options)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Survived column missing"**
   - Ensure your CSV has a 'Survived' column with 0/1 values

2. **Empty visualizations**
   - Check if filters are too restrictive
   - Verify data types are correct

3. **Slow performance**
   - Consider sampling large datasets
   - Use the `@st.cache_data` decorator for heavy computations

## 📈 Performance Tips

- Enable caching for data loading: `@st.cache_data`
- Filter data before complex visualizations
- Use `use_container_width=True` for responsive plots
- Close matplotlib figures to prevent memory leaks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Original Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data)
- Streamlit team for the amazing framework
- Plotly, Seaborn, and Altair communities for excellent visualization libraries



---

**Built with ❤️ using Streamlit**

*Transform data into insights with interactive visualizations!*
