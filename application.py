import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


class HousePricePredictorStreamlit:
    def __init__(self):
        self.model = None
        self.df = None
        self.model_trained = False
        
    def load_data(self, uploaded_file=None):
        """Load dataset from uploaded file or create sample data"""
        if uploaded_file is not None:
            try:
                self.df = pd.read_csv(uploaded_file)
                st.success(f"✅ Dataset loaded successfully with {len(self.df)} records!")
                return True
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                return False
        else:
            # Create sample data for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            self.df = pd.DataFrame({
                'beds': np.random.randint(1, 6, n_samples),
                'baths': np.random.randint(1, 4, n_samples),
                'size': np.random.randint(800, 4000, n_samples),
            })
            
            # Create realistic price based on features with some noise
            self.df['price'] = (
                self.df['beds'] * 45000 + 
                self.df['baths'] * 35000 + 
                self.df['size'] * 120 + 
                np.random.normal(50000, 30000, n_samples)
            )
            
            # Ensure no negative prices
            self.df['price'] = np.maximum(self.df['price'], 50000)
            
            st.info("📊 Using sample dataset for demonstration (1000 records)")
            return True
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        if self.df is None:
            return False
            
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        
        if len(self.df) < initial_rows:
            st.warning(f"⚠️ Removed {initial_rows - len(self.df)} rows with missing values")
        
        return True
    
    def create_visualizations(self):
        """Create interactive visualizations using Plotly"""
        if self.df is None:
            st.error("No data available for visualization!")
            return
        
        st.header("📊 Data Visualization")
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Houses", len(self.df))
        with col2:
            st.metric("Avg Price", f"${self.df['price'].mean():,.0f}")
        with col3:
            st.metric("Avg Size", f"{self.df['size'].mean():,.0f} sq ft")
        with col4:
            st.metric("Price Range", f"${self.df['price'].min():,.0f} - ${self.df['price'].max():,.0f}")
        
        # Feature distributions
        st.subheader("🔍 Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram for price
            fig_price = px.histogram(
                self.df, x='price', 
                title='Price Distribution',
                labels={'price': 'Price ($)', 'count': 'Frequency'},
                nbins=30
            )
            fig_price.update_layout(showlegend=False)
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Histogram for size
            fig_size = px.histogram(
                self.df, x='size', 
                title='House Size Distribution',
                labels={'size': 'Size (sq ft)', 'count': 'Frequency'},
                nbins=30
            )
            fig_size.update_layout(showlegend=False)
            st.plotly_chart(fig_size, use_container_width=True)
        
        with col2:
            # Bar chart for bedrooms
            bed_counts = self.df['beds'].value_counts().sort_index()
            fig_beds = px.bar(
                x=bed_counts.index, y=bed_counts.values,
                title='Houses by Number of Bedrooms',
                labels={'x': 'Bedrooms', 'y': 'Count'}
            )
            st.plotly_chart(fig_beds, use_container_width=True)
            
            # Bar chart for bathrooms
            bath_counts = self.df['baths'].value_counts().sort_index()
            fig_baths = px.bar(
                x=bath_counts.index, y=bath_counts.values,
                title='Houses by Number of Bathrooms',
                labels={'x': 'Bathrooms', 'y': 'Count'}
            )
            st.plotly_chart(fig_baths, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🌡️ Feature Correlation Heatmap")
        corr_matrix = self.df[['beds', 'baths', 'size', 'price']].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter plots
        st.subheader("📈 Price vs Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_beds_price = px.scatter(
                self.df, x='beds', y='price',
                title='Price vs Bedrooms',
                labels={'beds': 'Bedrooms', 'price': 'Price ($)'},
                opacity=0.6
            )
            st.plotly_chart(fig_beds_price, use_container_width=True)
        
        with col2:
            fig_baths_price = px.scatter(
                self.df, x='baths', y='price',
                title='Price vs Bathrooms',
                labels={'baths': 'Bathrooms', 'price': 'Price ($)'},
                opacity=0.6
            )
            st.plotly_chart(fig_baths_price, use_container_width=True)
        
        with col3:
            fig_size_price = px.scatter(
                self.df, x='size', y='price',
                title='Price vs Size',
                labels={'size': 'Size (sq ft)', 'price': 'Price ($)'},
                opacity=0.6
            )
            st.plotly_chart(fig_size_price, use_container_width=True)
    
    def train_model(self, model_type="Linear Regression"):
        """Train the machine learning model"""
        if self.df is None:
            st.error("No data available for training!")
            return False
            
        # Check required columns
        required_features = ['beds', 'baths', 'size']
        if not all(col in self.df.columns for col in required_features + ['price']):
            st.error(f"Missing required columns. Expected: {required_features + ['price']}")
            st.error(f"Available columns: {list(self.df.columns)}")
            return False
        
        X = self.df[required_features]
        y = self.df['price']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model based on selection
        if model_type == "Linear Regression":
            self.model = LinearRegression()
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"${rmse:,.0f}")
        with col3:
            st.metric("MSE", f"${mse:,.0f}")
        
        # Feature importance for Linear Regression
        if model_type == "Linear Regression":
            st.subheader("📊 Feature Coefficients")
            coef_df = pd.DataFrame({
                'Feature': required_features,
                'Coefficient': self.model.coef_
            })
            
            fig_coef = px.bar(
                coef_df, x='Feature', y='Coefficient',
                title='Feature Coefficients (Impact on Price)',
                labels={'Coefficient': 'Price Impact ($)'}
            )
            st.plotly_chart(fig_coef, use_container_width=True)
            
            st.write(f"**Intercept:** ${self.model.intercept_:,.2f}")
        
        # Feature importance for Random Forest
        else:
            st.subheader("📊 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': required_features,
                'Importance': self.model.feature_importances_
            })
            
            fig_imp = px.bar(
                importance_df, x='Feature', y='Importance',
                title='Feature Importance',
                labels={'Importance': 'Importance Score'}
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        self.model_trained = True
        return True
    
    def predict_price(self, beds, baths, size):
        """Predict house price for given features"""
        if not self.model_trained or self.model is None:
            return None
            
        prediction_data = np.array([[beds, baths, size]])
        predicted_price = self.model.predict(prediction_data)[0]
        return predicted_price


def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🏠 House Price Prediction System")
    st.markdown("### Predict house prices using machine learning with interactive visualizations")
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = HousePricePredictorStreamlit()
    
    predictor = st.session_state.predictor
    
    # Sidebar for data loading and model configuration
    st.sidebar.header("🔧 Configuration")
    
    # File upload
    st.sidebar.subheader("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file", 
        type=['csv'],
        help="CSV should have columns: beds, baths, size, price"
    )
    
    # Load data button
    if st.sidebar.button("🔄 Load Data"):
        predictor.load_data(uploaded_file)
        predictor.preprocess_data()
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose Model Type",
        ["Linear Regression", "Random Forest"],
        help="Linear Regression is faster, Random Forest is more accurate"
    )
    
    # Main content
    if predictor.df is not None:
        # Data overview
        st.header("📋 Dataset Overview")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(predictor.df.head(10), use_container_width=True)
        with col2:
            st.subheader("Dataset Statistics")
            st.write(predictor.df.describe())
        
        # Visualizations
        predictor.create_visualizations()
        
        # Model training
        st.header("🤖 Model Training")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                if predictor.train_model(model_type):
                    st.success("✅ Model trained successfully!")
                else:
                    st.error("❌ Failed to train model!")
        
        # Prediction interface
        if predictor.model_trained:
            st.header("🔮 House Price Prediction")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Enter House Details")
                
                beds = st.number_input(
                    "Number of Bedrooms", 
                    min_value=1, max_value=10, value=3,
                    help="Enter the number of bedrooms"
                )
                
                baths = st.number_input(
                    "Number of Bathrooms", 
                    min_value=1.0, max_value=10.0, value=2.0, step=0.5,
                    help="Enter the number of bathrooms"
                )
                
                size = st.number_input(
                    "House Size (sq ft)", 
                    min_value=500, max_value=10000, value=1500,
                    help="Enter the house size in square feet"
                )
                
                if st.button("💰 Predict Price", type="primary"):
                    predicted_price = predictor.predict_price(beds, baths, size)
                    
                    if predicted_price is not None:
                        st.session_state.prediction_result = {
                            'beds': beds,
                            'baths': baths,
                            'size': size,
                            'predicted_price': predicted_price
                        }
            
            with col2:
                st.subheader("Prediction Result")
                
                if 'prediction_result' in st.session_state:
                    result = st.session_state.prediction_result
                    
                    # Display prediction in a nice format
                    st.markdown("### 🏠 House Details")
                    st.write(f"🛏️ **Bedrooms:** {result['beds']}")
                    st.write(f"🚿 **Bathrooms:** {result['baths']}")
                    st.write(f"📐 **Size:** {result['size']:,} sq ft")
                    
                    st.markdown("### 💰 Predicted Price")
                    st.success(f"**${result['predicted_price']:,.2f}**")
                    
                    # Price per square foot
                    price_per_sqft = result['predicted_price'] / result['size']
                    st.write(f"📊 **Price per sq ft:** ${price_per_sqft:.2f}")
                else:
                    st.info("👆 Enter house details and click 'Predict Price' to see results")
    
    else:
        st.info("📤 Please upload a CSV file or click 'Load Data' to use sample data")
        
        st.markdown("""
        ### 📋 Expected CSV Format:
        Your CSV file should have the following columns:
        - **beds**: Number of bedrooms (integer)
        - **baths**: Number of bathrooms (float)
        - **size**: House size in square feet (integer)
        - **price**: House price in dollars (float)
        
        ### 📊 Sample Data Available:
        If you don't have a dataset, click 'Load Data' without uploading a file to use our sample dataset with 1000 house records.
        """)


if __name__ == "__main__":
    main()
