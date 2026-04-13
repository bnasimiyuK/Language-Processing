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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Language Identifier Pro", page_icon="🌍", layout="wide")

# --- 2. THE FIX: RELOAD & BALANCE DATA ---
@st.cache_resource
def load_and_train_model():
    try:
        # Load CSV
        df = pd.read_csv('labeled_data.csv')
        df['language'] = df['language'].str.strip().str.capitalize()

        # CLEANING
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[.,!?#@$%^&*()_+=\[\]{};:"\'<>/\\|`~]', '', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text) 
            return re.sub(r'\s+', ' ', text).strip()

        df['cleaned'] = df['text'].apply(clean_text)
        df = df[df['cleaned'].str.len() > 2] # Ignore very short artifacts

        # THE FIX: BALANCING DATA (Ensures the model doesn't favor the largest category)
        min_size = df['language'].value_counts().min()
        df = df.groupby('language').sample(n=min_size, random_state=42)

        # FEATURE EXTRACTION (Word + Char combined)
        # We use Char (2-5) for Lubukusu/Sheng and Word (1-1) for English/Swahili
        tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=15000)
        X = tfidf.fit_transform(df['cleaned'])
        
        le = LabelEncoder()
        y = le.fit_transform(df['language'])

        # TRAIN/TEST
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # MODELS (Class Weight 'balanced' helps Logistic Regression)
        nb = MultinomialNB(alpha=0.01)
        lr = LogisticRegression(max_iter=2000, class_weight='balanced')
        
        # Train both models separately
        nb.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        
        # Calculate accuracies
        nb_acc = accuracy_score(y_test, nb.predict(X_test))
        lr_acc = accuracy_score(y_test, lr.predict(X_test))
        
        # Get confusion matrices
        nb_cm = confusion_matrix(y_test, nb.predict(X_test))
        lr_cm = confusion_matrix(y_test, lr.predict(X_test))
        
        # Final fit on full dataset
        nb.fit(X, y)
        lr.fit(X, y)
        
        return tfidf, nb, lr, le, clean_text, nb_acc, lr_acc, nb_cm, lr_cm, df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None, None, None, None

# --- INITIALIZE ---
# IMPORTANT: Use st.button to clear cache if you added new data to CSV
if st.sidebar.button("🔄 Refresh Data/Reload CSV"):
    st.cache_resource.clear()

tfidf, nb_model, lr_model, le, clean_func, nb_accuracy, lr_accuracy, nb_cm, lr_cm, raw_df = load_and_train_model()

if tfidf is not None:
    # --- UI ---
    st.title("🌍 Language Identification System")
    
    # Model selection
    model_choice = st.sidebar.radio(
        "Select Model:",
        ["Naive Bayes", "Logistic Regression"]
    )
    
    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area("Enter Text:", height=150)
        
        # Display accuracy for selected model
        if model_choice == "Naive Bayes":
            current_model = nb_model
            current_accuracy = nb_accuracy
            st.sidebar.metric("Naive Bayes Accuracy", f"{current_accuracy*100:.1f}%")
        else:
            current_model = lr_model
            current_accuracy = lr_accuracy
            st.sidebar.metric("Logistic Regression Accuracy", f"{current_accuracy*100:.1f}%")
        
        if st.button("Predict", type="primary"):
            cleaned = clean_func(user_input)
            vec = tfidf.transform([cleaned])
            
            probs = current_model.predict_proba(vec)[0]
            pred_idx = probs.argmax()
            
            lang = le.inverse_transform([pred_idx])[0]
            conf = probs[pred_idx] * 100

            st.success(f"### Predicted: {lang} ({conf:.1f}% Confidence)")
            
            # Probability Bars
            st.subheader("Class Probabilities")
            for l, p in zip(le.classes_, probs):
                st.write(f"**{l}**")
                st.progress(float(p))

    with col2:
        st.subheader("Language Distribution (Balanced)")
        st.bar_chart(raw_df['language'].value_counts())
        
        # Show confusion matrix for selected model
        with st.expander("Show Confusion Heatmap"):
            fig, ax = plt.subplots()
            if model_choice == "Naive Bayes":
                disp = ConfusionMatrixDisplay(confusion_matrix=nb_cm, display_labels=le.classes_)
            else:
                disp = ConfusionMatrixDisplay(confusion_matrix=lr_cm, display_labels=le.classes_)
            disp.plot(cmap='Blues', ax=ax, colorbar=False)
            st.pyplot(fig)
            
        # Model comparison
        with st.expander("Model Comparison"):
            comparison_data = {
                "Model": ["Naive Bayes", "Logistic Regression"],
                "Accuracy": [f"{nb_accuracy*100:.1f}%", f"{lr_accuracy*100:.1f}%"]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)