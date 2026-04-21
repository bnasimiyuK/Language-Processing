import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Language Identifier Pro", page_icon="🌍", layout="wide")

# Initialize Session State for toggling details
if 'show_details' not in st.session_state:
    st.session_state.show_details = False

# --- 2. MODEL TRAINING (With Calibration) ---
@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv('labeled_data.csv')
        df['language'] = df['language'].str.strip().str.capitalize()

        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[.,!?#@$%^&*()_+=\[\]{};:"\'<>/\\|`~]', '', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text) 
            return re.sub(r'\s+', ' ', text).strip()

        df['cleaned'] = df['text'].apply(clean_text)
        df = df[df['cleaned'].str.len() > 2] 

        min_size = df['language'].value_counts().min()
        df = df.groupby('language').sample(n=min_size, random_state=42)

        tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=15000)
        X = tfidf.fit_transform(df['cleaned'])
        
        le = LabelEncoder()
        y = le.fit_transform(df['language'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # CALIBRATED MODELS: This fixes the confidence gap (99% vs 34%)
        base_nb = MultinomialNB(alpha=0.1)
        nb = CalibratedClassifierCV(base_nb, cv=5, method='sigmoid')
        
        base_lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.5)
        lr = CalibratedClassifierCV(base_lr, cv=5, method='sigmoid')
        
        nb.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        
        nb_acc = accuracy_score(y_test, nb.predict(X_test))
        lr_acc = accuracy_score(y_test, lr.predict(X_test))
        nb_cm = confusion_matrix(y_test, nb.predict(X_test))
        lr_cm = confusion_matrix(y_test, lr.predict(X_test))
        
        return tfidf, nb, lr, le, clean_text, nb_acc, lr_acc, nb_cm, lr_cm, df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None, None, None, None

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Settings & Tools")

# Refresh Button
if st.sidebar.button("🔄 Refresh Data/Models"):
    st.cache_resource.clear()
    st.session_state.show_details = False
    st.rerun()

# Toggle Button (placed in sidebar)
if not st.session_state.show_details:
    if st.sidebar.button("📊 Show Model Analytics"):
        st.session_state.show_details = True
        st.rerun()
else:
    if st.sidebar.button("🙈 Hide Model Analytics"):
        st.session_state.show_details = False
        st.rerun()

tfidf, nb_model, lr_model, le, clean_func, nb_accuracy, lr_accuracy, nb_cm, lr_cm, raw_df = load_and_train_model()

if tfidf is not None:
    # Model Selection in Sidebar
    model_choice = st.sidebar.radio("Select Active Model:", ["Naive Bayes", "Logistic Regression"])
    
    # --- MAIN UI ---
    st.title("🌍 Language Identification System")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area("Enter Text:", height=150)
        
        # Determine active model data
        if model_choice == "Naive Bayes":
            st.sidebar.metric("NB Accuracy", f"{nb_accuracy*100:.1f}%")
            current_model = nb_model
            current_cm = nb_cm
        else:
            st.sidebar.metric("LR Accuracy", f"{lr_accuracy*100:.1f}%")
            current_model = lr_model
            current_cm = lr_cm
        
        if st.button("Predict Language", type="primary"):
            if user_input.strip():
                cleaned = clean_func(user_input)
                vec = tfidf.transform([cleaned])
                probs = current_model.predict_proba(vec)[0]
                pred_idx = probs.argmax()
                lang = le.inverse_transform([pred_idx])[0]
                conf = probs[pred_idx] * 100

                st.success(f"### Predicted: {lang} ({conf:.1f}% Confidence)")
                
                # Probabilities show here ONLY if enabled via Sidebar
                if st.session_state.show_details:
                    st.write("---")
                    st.subheader("Confidence Breakdown")
                    for l, p in zip(le.classes_, probs):
                        st.write(f"**{l}** ({p*100:.1f}%)")
                        st.progress(float(p))
            else:
                st.warning("Please enter text first.")

    with col2:
        # Analytics Section - Hidden until Sidebar button is clicked
        if st.session_state.show_details:
            st.subheader("Data & Model Analytics")
            
            st.write("**Language Distribution**")
            st.bar_chart(raw_df['language'].value_counts())
            
            with st.expander("Confusion Matrix Heatmap", expanded=True):
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=current_cm, display_labels=le.classes_)
                disp.plot(cmap='Blues', ax=ax, colorbar=False)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
            with st.expander("Model Accuracy Comparison"):
                st.table(pd.DataFrame({
                    "Model": ["Naive Bayes", "Logistic Regression"],
                    "Accuracy": [f"{nb_accuracy*100:.1f}%", f"{lr_accuracy*100:.1f}%"]
                }))
        