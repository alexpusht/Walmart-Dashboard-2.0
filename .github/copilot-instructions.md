# Walmart Sales Dashboard - AI Agent Instructions

## Overview
This is a single-file Streamlit application (`DashboardWalmart.py`) that creates an interactive sales dashboard for Walmart transaction data. It showcases data visualization, filtering, KPIs, forecasting, and customer segmentation capabilities.

## Architecture
- **Single-file design**: All logic in `DashboardWalmart.py` for simplicity as a portfolio project
- **Data flow**: CSV → pandas DataFrame → filtered views → calculations/visualizations
- **Components**: Sidebar filters, KPI metrics with growth comparisons, multiple Plotly charts, ML-based forecasting and segmentation

## Key Workflows
- **Run app**: `streamlit run DashboardWalmart.py` (assumes `Walmart.csv` in same directory)
- **Data loading**: Uses `@st.cache_data` for performance
- **Filtering**: Multi-select filters in sidebar affect all downstream calculations

## Data Handling
- **Core columns**: `transaction_date`, `category`, `store_location`, `customer_loyalty_level`, `quantity_sold`, `unit_price`
- **Date processing**: Convert to datetime, filter by date ranges
- **Sales calculation**: Always `quantity_sold * unit_price` for revenue
- **Growth metrics**: Compare current period vs previous 30 days

## UI Conventions
- **Language**: Dutch text for all UI elements (titles, labels, buttons)
- **Currency**: Euro (€) with comma separators (e.g., `€{value:,.2f}`)
- **Layout**: Wide layout, expanded sidebar, columns for KPIs/charts
- **Colors**: Consistent color schemes per chart type (Blues for categories, Greens for locations)

## Analysis Patterns
- **Forecasting**: Linear regression on 7-day moving average of last 60 days, weighted with 4-week historical average
- **Segmentation**: K-means clustering (n=3) on standardized age/income data
- **Recommendations**: Dynamic business insights based on top performers, loyalty analysis, promotion impact
- **Visualizations**: Plotly Express for bars/pies, Plotly Graph Objects for complex forecasts

## Dependencies
- **Core**: streamlit, pandas, numpy
- **Visualization**: plotly (express + graph_objects)
- **ML**: scikit-learn (LinearRegression, StandardScaler, KMeans)

## Key Files
- `DashboardWalmart.py`: Main application
- `Walmart.csv`: Transaction data (assumed present)

When modifying, maintain the single-file structure and Dutch UI language. Test visualizations with sample data before deploying.