import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time

# Must be first Streamlit command
st.set_page_config(
    page_title="Spam Detector Pro",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# Initialize NLTK
@st.cache_resource
def load_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    return True


load_nltk()

# Text preprocessing
ps = PorterStemmer()


def text_transform(text):
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in set(stopwords.words('english')) | set(string.punctuation)]
    stem = [ps.stem(t) for t in tokens]
    return " ".join(stem)


# Load models
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('TfisfVectorize.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except:
        return None, None


tfidf, model = load_models()

# Modern, clean CSS design
st.markdown("""
<style>
    /* Reset and base styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* Clean white background with subtle gray */
    .stApp {
        background-color: #fafafa;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }

    /* Main container - centered card */
    .main-card {
        background: white;
        max-width: 700px;
        margin: 2rem auto;
        padding: 3rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
    }

    /* Header section */
    .header {
        text-align: center;
        margin-bottom: 2.5rem;
    }

    .logo {
        width: 64px;
        height: 64px;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border-radius: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1.5rem;
        font-size: 2rem;
    }

    .title {
        font-size: 1.875rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Input section */
    .input-wrapper {
        margin-bottom: 1.5rem;
    }

    .input-label {
        display: block;
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.5rem;
    }

    /* Custom textarea styling */
    .stTextArea textarea {
        min-height: 140px !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        color: #111827 !important;
        background-color: #fff !important;
        transition: all 0.2s !important;
        resize: vertical !important;
    }

    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        outline: none !important;
    }

    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
    }

    /* Button styling */
    .stButton button {
        width: 100% !important;
        background-color: #3b82f6 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.875rem 1.5rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }

    .stButton button:hover {
        background-color: #2563eb !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }

    .stButton button:active {
        transform: translateY(0);
    }

    /* Results section */
    .result-card {
        margin-top: 2rem;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid;
        animation: fadeIn 0.3s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-spam {
        background-color: #fef2f2;
        border-left-color: #ef4444;
    }

    .result-ham {
        background-color: #f0fdf4;
        border-left-color: #22c55e;
    }

    .result-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .result-icon {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }

    .result-spam .result-icon {
        background-color: #fee2e2;
    }

    .result-ham .result-icon {
        background-color: #dcfce7;
    }

    .result-title {
        font-size: 1.125rem;
        font-weight: 600;
    }

    .result-spam .result-title {
        color: #991b1b;
    }

    .result-ham .result-title {
        color: #166534;
    }

    .result-desc {
        color: #6b7280;
        font-size: 0.875rem;
        margin-left: 2.75rem;
    }

    /* Stats bar */
    .stats {
        display: flex;
        gap: 2rem;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid #e5e7eb;
    }

    .stat {
        text-align: center;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid #e5e7eb;
        color: #9ca3af;
        font-size: 0.875rem;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Loading spinner */
    .stSpinner > div {
        border-color: #3b82f6 !important;
    }

    /* Alert styling */
    .stAlert {
        background-color: #fef3c7;
        border: 1px solid #fcd34d;
        color: #92400e;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Main card container
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <div class="logo">🛡️</div>
    <h1 class="title">Spam Detector Pro</h1>
    <p class="subtitle">AI-powered SMS and email classification</p>
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
st.markdown('<label class="input-label">Enter message to analyze</label>', unsafe_allow_html=True)

input_sms = st.text_area(
    "",
    placeholder="Paste your message here...",
    height=140,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("Analyze Message", type="primary", use_container_width=True)

# Analysis logic
if analyze:
    if not input_sms.strip():
        st.warning("Please enter a message to analyze")
    else:
        if tfidf is None or model is None:
            st.error("Model files not found. Please check your installation.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(0.5)  # Small delay for UX

                transformed = text_transform(input_sms)
                vector = tfidf.transform([transformed])
                prediction = model.predict(vector)[0]

                # Add to history
                st.session_state.history.insert(0, {
                    'text': input_sms[:60] + "..." if len(input_sms) > 60 else input_sms,
                    'result': prediction
                })

                # Display result
                if prediction == 1:
                    st.markdown("""
                    <div class="result-card result-spam">
                        <div class="result-header">
                            <div class="result-icon">🚨</div>
                            <div class="result-title">Spam Detected</div>
                        </div>
                        <div class="result-desc">This message contains patterns commonly associated with spam or phishing attempts.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-card result-ham">
                        <div class="result-header">
                            <div class="result-icon">✓</div>
                            <div class="result-title">Message is Safe</div>
                        </div>
                        <div class="result-desc">No suspicious patterns detected. This appears to be a legitimate message.</div>
                    </div>
                    """, unsafe_allow_html=True)

# Statistics
total = len(st.session_state.history)
spam_count = sum(1 for h in st.session_state.history if h['result'] == 1)
safe_count = total - spam_count

st.markdown(f"""
<div class="stats">
    <div class="stat">
        <div class="stat-value">{total}</div>
        <div class="stat-label">Analyzed</div>
    </div>
    <div class="stat">
        <div class="stat-value" style="color: #ef4444;">{spam_count}</div>
        <div class="stat-label">Spam</div>
    </div>
    <div class="stat">
        <div class="stat-value" style="color: #22c55e;">{safe_count}</div>
        <div class="stat-label">Safe</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Recent history
if st.session_state.history:
    with st.expander("Recent scans"):
        for i, item in enumerate(st.session_state.history[:3]):
            icon = "🚨" if item['result'] == 1 else "✓"
            color = "#ef4444" if item['result'] == 1 else "#22c55e"
            label = "Spam" if item['result'] == 1 else "Safe"

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: #f9fafb; border-radius: 6px; margin-bottom: 0.5rem;">
                <span style="font-size: 0.875rem; color: #374151; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 400px;">{icon} {item['text']}</span>
                <span style="font-size: 0.75rem; font-weight: 600; color: {color}; background: white; padding: 0.25rem 0.75rem; border-radius: 12px; border: 1px solid {color}20;">{label}</span>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Secure local processing • No data stored
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)