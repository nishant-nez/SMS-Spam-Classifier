import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import plotly.express as px
import pandas as pd
import time

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt_tab')
    nltk.download('stopwords')

download_nltk_data()

# Initialize components
ps = PorterStemmer()

@st.cache_resource
def load_models():
    tfidf = pickle.load(open('./models/vectorizer.pkl', 'rb'))
    model = pickle.load(open('./models/model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_models()

def transform_text(text):
    """Transform text for classification"""
    # Lowercase & tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    # Filter out non-alphanumeric tokens, stopwords, and punctuation, then perform stemming
    cleaned_tokens = [
        ps.stem(token) for token in tokens
        if token.isalnum() and token not in stopwords.words('english') and token not in string.punctuation
    ]
    
    return " ".join(cleaned_tokens)

def get_prediction_confidence(vector_input):
    """Get prediction probabilities"""
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(vector_input)[0]
            return {
                'Not Spam': probabilities[0],
                'Spam': probabilities[1]
            }
    except:
        pass
    return None

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-spam {
        background: linear-gradient(90deg, #ff6b6b, #ff5722);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .result-not-spam {
        background: linear-gradient(90deg, #4caf50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-box {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: #212426;
        border: 1px solid #bbdefb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .github-button {
        background: #333;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 About This App")
    st.markdown("""
    This SMS Spam Classifier uses machine learning to detect whether a text message is spam or legitimate (ham).
    
    **Key Features:**
    - Real-time SMS classification
    - Confidence scores
    - Text preprocessing visualization
    - Dataset information
    """)
    
    st.markdown("## 🔗 Connect")
    github_url = st.text_input("GitHub Repository URL:", placeholder="https://github.com/nishant-nez/SMS-Spam-Classifier.git")
    
    if github_url:
        st.markdown(f'<a href="{github_url}" class="github-button" target="_blank">🔗 View on GitHub</a>', unsafe_allow_html=True)
    
    st.markdown("## 📈 Model Info")
    with st.expander("Technical Details"):
        st.markdown("""
        **Preprocessing Steps:**
        1. Text lowercasing
        2. Tokenization
        3. Removal of stopwords
        4. Removal of punctuation
        5. Porter stemming
        6. TF-IDF vectorization
        
        **Model:** Pre-trained classifier
        **Features:** TF-IDF vectors
        """)
    
    # Dataset Statistics
    st.markdown("## 📊 Dataset Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", "5,574")
    with col2:
        st.metric("Spam Rate", "13.4%")

# Main content
st.markdown('<h1 class="main-header">📱 SMS Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect spam messages using advanced machine learning</p>', unsafe_allow_html=True)

# Main input section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Enter Your SMS Message")
    input_sms = st.text_area(
        "Message Content:",
        height=120,
        placeholder="Type or paste your SMS message here...",
        help="Enter the SMS message you want to classify as spam or not spam."
    )
    
    # Example messages
    st.markdown("**Quick Examples:**")
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        if st.button("📢 Try Spam Example", use_container_width=True):
            input_sms = "URGENT! You've won £1000 cash! Call 09061701461 now! Claims cost 25p/min"
            st.rerun()
    
    with example_col2:
        if st.button("✅ Try Ham Example", use_container_width=True):
            input_sms = "Hi mom, I'll be home for dinner around 7pm. See you soon!"
            st.rerun()

with col2:
    st.markdown("### 🎯 Classification Result")
    
    if st.button("🔍 Classify Message", type="primary", use_container_width=True):
        if input_sms.strip():
            # Show processing
            with st.spinner("Analyzing message..."):
                time.sleep(0.5)  # Small delay for better UX
                
                # Process the message
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                confidence_scores = get_prediction_confidence(vector_input)
                
                # Display result
                if result == 1:
                    st.markdown('<div class="result-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                    st.error("This message appears to be spam!")
                else:
                    st.markdown('<div class="result-not-spam">✅ LEGITIMATE MESSAGE</div>', unsafe_allow_html=True)
                    st.success("This message appears to be legitimate!")
                
                # Show confidence scores if available
                if confidence_scores:
                    st.markdown("**Confidence Scores:**")
                    
                    # Create a simple bar chart
                    conf_df = pd.DataFrame([
                        {"Category": "Not Spam", "Confidence": confidence_scores['Not Spam']},
                        {"Category": "Spam", "Confidence": confidence_scores['Spam']}
                    ])
                    
                    fig = px.bar(
                        conf_df, 
                        x='Category', 
                        y='Confidence',
                        color='Category',
                        color_discrete_map={'Not Spam': '#4caf50', 'Spam': '#ff6b6b'},
                        title="Classification Confidence"
                    )
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show numerical confidence
                    st.metric(
                        "Confidence Score", 
                        f"{max(confidence_scores.values()):.1%}",
                        help="How confident the model is in its prediction"
                    )
        else:
            st.warning("Please enter a message to classify!")

# Text Analysis Section
if input_sms.strip():
    st.markdown("---")
    st.markdown("## 🔍 Text Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Original Message")
        st.text_area("Raw Input:", input_sms, height=100, disabled=True)
        
        # Message statistics
        st.markdown("**Message Statistics:**")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Characters", len(input_sms))
        with stats_col2:
            st.metric("Words", len(input_sms.split()))
        with stats_col3:
            st.metric("Sentences", len([s for s in input_sms.split('.') if s.strip()]))
    
    with col2:
        st.markdown("### Processed Message")
        transformed_sms = transform_text(input_sms)
        st.text_area("After Preprocessing:", transformed_sms, height=100, disabled=True)
        
        # Processing info
        st.markdown("**Processing Applied:**")
        st.markdown("""
        - ✅ Lowercased
        - ✅ Tokenized
        - ✅ Removed stopwords
        - ✅ Removed punctuation
        - ✅ Applied stemming
        """)

# Information Section
st.markdown("---")
st.markdown("## 📚 About the SMS Spam Collection Dataset")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    <div class="info-box">
    <h4>📊 Dataset Overview</h4>
    <p>The <strong>SMS Spam Collection Dataset</strong> is a public collection of SMS messages tagged as either spam or legitimate (ham). This dataset has been widely used for text classification research.</p>
    
    <ul>
    <li><strong>Total Messages:</strong> 5,574</li>
    <li><strong>Spam Messages:</strong> 747 (13.4%)</li>
    <li><strong>Ham Messages:</strong> 4,827 (86.6%)</li>
    <li><strong>Source:</strong> UCI Machine Learning Repository</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="info-box">
    <h4>🔬 Model Performance</h4>
    <p>The classifier has been trained using advanced machine learning techniques with the following characteristics:</p>
    
    <ul>
    <li><strong>Feature Extraction:</strong> TF-IDF Vectorization</li>
    <li><strong>Text Processing:</strong> NLTK with Porter Stemming</li>
    <li><strong>Preprocessing:</strong> Stopword removal, punctuation filtering</li>
    <li><strong>Application:</strong> Real-time SMS classification</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("### 🚀 Features of This App")

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown("""
    **🤖 AI-Powered**
    - Advanced ML classification
    - Real-time processing
    - High accuracy detection
    """)

with feature_col2:
    st.markdown("""
    **📊 Detailed Analysis**
    - Confidence scores
    - Text preprocessing view
    - Message statistics
    """)

with feature_col3:
    st.markdown("""
    **🎨 User-Friendly**
    - Beautiful interface
    - Example messages
    - Interactive visualizations
    """)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ using Streamlit | SMS Spam Classification powered by Machine Learning
    </div>
    """, 
    unsafe_allow_html=True
)