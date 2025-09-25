import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(90deg, #ff6b6b 0%, #ffa726 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        data = pd.read_csv('datasets/bin_data.csv')
        if 'Unnamed: 0' in data.columns:
            data = data.drop('Unnamed: 0', axis=1)
        
        # Use only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data = numeric_data.dropna()
        
        return numeric_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Create sample test data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic network features
    data = {
        'duration': np.random.exponential(30, n_samples),
        'src_bytes': np.random.exponential(1000, n_samples),
        'dst_bytes': np.random.exponential(500, n_samples),
        'count': np.random.poisson(10, n_samples),
        'srv_count': np.random.poisson(5, n_samples),
        'serror_rate': np.random.beta(1, 10, n_samples),
        'srv_serror_rate': np.random.beta(1, 8, n_samples),
        'same_srv_rate': np.random.beta(5, 2, n_samples),
        'diff_srv_rate': np.random.beta(1, 5, n_samples),
        'dst_host_count': np.random.poisson(50, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic labels (0 = normal, 1 = attack)
    # Add some correlation with features to make it realistic
    attack_prob = (
        0.1 +  # base probability
        0.3 * (df['serror_rate'] > 0.5) +  # high error rate indicates attack
        0.2 * (df['count'] > 20) +  # high connection count
        0.2 * (df['same_srv_rate'] < 0.3)  # low same service rate
    )
    
    df['intrusion'] = np.random.binomial(1, np.clip(attack_prob, 0, 0.8), n_samples)
    
    return df

def create_data_overview(df):
    """Create data overview visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Attack distribution
        attack_counts = df['intrusion'].value_counts()
        fig_pie = px.pie(
            values=attack_counts.values,
            names=['Normal Traffic', 'Attack Traffic'],
            title="Network Traffic Distribution",
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Feature correlation heatmap
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results"""
    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='linear', probability=True),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
    }
    
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': predictions,
            'training_time': training_time
        }
        
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text('Training complete!')
    return results

def create_results_dashboard(results, y_test):
    """Create comprehensive results dashboard"""
    
    # Model comparison
    st.subheader("🏆 Model Performance Comparison")
    
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in model_names]
    training_times = [results[name]['training_time'] for name in model_names]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig_acc = px.bar(
            x=model_names,
            y=accuracies,
            title="Model Accuracy Comparison",
            labels={'x': 'Model', 'y': 'Accuracy (%)'},
            color=accuracies,
            color_continuous_scale='Viridis'
        )
        fig_acc.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # Training time comparison
        fig_time = px.bar(
            x=model_names,
            y=training_times,
            title="Training Time Comparison",
            labels={'x': 'Model', 'y': 'Training Time (seconds)'},
            color=training_times,
            color_continuous_scale='Plasma'
        )
        fig_time.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Best model analysis
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]
    
    st.subheader(f"🥇 Best Model: {best_model_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="success-card">
            <h3>Accuracy</h3>
            <h2>{best_model['accuracy']*100:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        correct_predictions = int(best_model['accuracy'] * len(y_test))
        st.markdown(f"""
        <div class="metric-card">
            <h3>Correct</h3>
            <h2>{correct_predictions}/{len(y_test)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Training Time</h3>
            <h2>{best_model['training_time']:.3f}s</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        attacks_detected = ((y_test == 1) & (best_model['predictions'] == 1)).sum()
        total_attacks = (y_test == 1).sum()
        st.markdown(f"""
        <div class="warning-card">
            <h3>Attacks Detected</h3>
            <h2>{attacks_detected}/{total_attacks}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Confusion Matrix
    st.subheader("🎯 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cm = confusion_matrix(y_test, best_model['predictions'])
        
        fig_cm = px.imshow(
            cm,
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=['Normal', 'Attack'],
            y=['Normal', 'Attack'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Classification report
        report = classification_report(
            y_test, best_model['predictions'],
            target_names=['Normal', 'Attack'],
            output_dict=True
        )
        
        metrics_df = pd.DataFrame({
            'Normal': [report['Normal']['precision'], report['Normal']['recall'], report['Normal']['f1-score']],
            'Attack': [report['Attack']['precision'], report['Attack']['recall'], report['Attack']['f1-score']]
        }, index=['Precision', 'Recall', 'F1-Score'])
        
        fig_metrics = px.bar(
            metrics_df,
            title="Classification Metrics",
            barmode='group',
            labels={'index': 'Metric', 'value': 'Score'}
        )
        fig_metrics.update_layout(height=400)
        st.plotly_chart(fig_metrics, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ AI Network Intrusion Detection System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Advanced AI-powered cybersecurity system for real-time network threat detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Demo Dataset (Recommended)", "NSL-KDD Dataset", "Upload Custom Data"]
    )
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", 1, 100, 42)
    
    # Load data based on selection
    if data_source == "Demo Dataset (Recommended)":
        df = load_sample_data()
        st.success("✅ Demo dataset loaded successfully!")
        
    elif data_source == "NSL-KDD Dataset":
        df = load_data()
        if df is not None:
            st.success("✅ NSL-KDD dataset loaded successfully!")
        else:
            st.error("❌ Could not load NSL-KDD dataset. Using demo data instead.")
            df = load_sample_data()
            
    else:  # Upload custom data
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ Custom dataset loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                df = load_sample_data()
        else:
            st.info("📁 Please upload a CSV file or use demo data")
            df = load_sample_data()
    
    if df is not None:
        # Data overview
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Connections", f"{len(df):,}")
        
        with col2:
            normal_count = (df['intrusion'] == 0).sum()
            st.metric("Normal Traffic", f"{normal_count:,}")
        
        with col3:
            attack_count = (df['intrusion'] == 1).sum()
            st.metric("Attack Traffic", f"{attack_count:,}")
        
        with col4:
            attack_percentage = (attack_count / len(df)) * 100
            st.metric("Attack Rate", f"{attack_percentage:.1f}%")
        
        # Data visualizations
        create_data_overview(df)
        
        # Sample data display
        st.subheader("📋 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Training section
        st.header("🤖 AI Model Training")
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            
            # Prepare data
            X = df.drop('intrusion', axis=1)
            y = df['intrusion']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            st.info(f"🔄 Training models on {len(X_train):,} samples, testing on {len(X_test):,} samples...")
            
            # Train models
            results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # Display results
            st.header("📈 Results Dashboard")
            create_results_dashboard(results, y_test)
            
            # Success message
            best_accuracy = max(result['accuracy'] for result in results.values())
            st.balloons()
            st.success(f"🎉 Training completed! Best accuracy: {best_accuracy*100:.2f}%")
            
            # Download results
            st.subheader("💾 Export Results")
            
            results_summary = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy (%)': [results[name]['accuracy'] * 100 for name in results.keys()],
                'Training Time (s)': [results[name]['training_time'] for name in results.keys()]
            })
            
            csv = results_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="intrusion_detection_results.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🛡️ Network Intrusion Detection System | Built with Streamlit & Scikit-learn</p>
        <p>Protecting networks with AI-powered cybersecurity</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()