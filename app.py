import streamlit as st
import pickle
import re
import nltk
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

# Download stopwords
nltk.download('stopwords')

# Load model components
model = pickle.load(open("naive_bayes_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Preprocess function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Streamlit page config
st.set_page_config(page_title="Flipkart Review Sentiment Classifier", page_icon="🛍️", layout="centered")

# 💅 Custom CSS Styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
            padding: 1rem;
        }
        .title {
            font-size: 2.2rem;
            font-weight: bold;
            color: #003566;
            text-align: center;
            margin-top: 0.5rem;
        }
        .stTextArea > label {
            font-size: 1.1rem;
            font-weight: 600;
        }
        .footer {
            font-size: 0.85rem;
            text-align: center;
            color: #6c757d;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# 🖼️ Logo
st.image("flipkart_logo.png", width=120)

# 📌 Title
st.markdown('<div class="title">🛍️ Flipkart Review Sentiment Classifier</div>', unsafe_allow_html=True)

# 📖 Description
st.markdown("""
Welcome to the Flipkart Review Sentiment Classifier!  
This app uses a machine learning model trained on real customer reviews to predict the sentiment as:
- ✅ **Positive**
- ❌ **Negative**
- ➖ **Neutral**
""")

# 📂 Sidebar
with st.sidebar:
    st.image("flipkart_logo.png", width=100)
    st.markdown("## 🔍 Navigation")
    st.markdown("- Home")
    st.markdown("- About Project")
    st.markdown("- Contact Us")
    st.markdown("---")
    st.markdown("👩‍💻 **Made with ❤️ by Vartika Singh**")
    # 📊 Sentiment Stats (Static Example)
    st.markdown("---")
    st.subheader("📊 Sentiment Stats")

    col1, col2 = st.columns([1.2, 0.8])  # Adjust width ratio

    with col1:
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [33.3, 33.3, 33.3]
        colors = ['#00b4d8', '#caf0f8', '#007f5f']

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
            textprops=dict(color="black", fontsize=10)
        )
        ax.axis('equal')
        st.pyplot(fig)

    with col2:
        st.markdown("""
        <div style="background-color: #007f5f; padding: 1rem; border-radius: 10px;">
            <p><strong>✅ Model:</strong> Naive Bayes</p>
            <p><strong>📂 Dataset:</strong> 5000 product reviews</p>
        </div>
        """, unsafe_allow_html=True)

# ✨ Sample review dropdown
st.markdown("### Try a Sample Review:")
sample_reviews = {
    "Excellent!": "Amazing phone. Performance and battery life are top-notch.",
    "Bad Product": "Stopped working within a week. Very disappointed.",
    "Okay-Okay": "Product is ok ok."
}
selected = st.selectbox("Choose a sample review (or enter your own below):", ["--"] + list(sample_reviews.keys()))

user_input = ""
if selected != "--":
    user_input = sample_reviews[selected]

# 📝 Text area for review
user_input = st.text_area("Enter your customer review here:", value=user_input, height=150)

# 🔍 Predict button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter or select a review to analyze.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned]).toarray()
        pred = model.predict(vec)
        label = label_encoder.inverse_transform(pred)[0]

        # Styled result
        if label == "positive":
            st.success("✅ Predicted Sentiment: **POSITIVE** 😊")
        elif label == "negative":
            st.error("❌ Predicted Sentiment: **NEGATIVE** 😞")
        else:
            st.info("➖ Predicted Sentiment: **NEUTRAL** 😐")



# 💚 Footer
st.markdown("""
<hr>
<div class='footer'>
📞 Contact: Vartika Singh | 📧 Email: vartika.singh.2004lko@gmail.com  
🔗 <a href="https://github.com/Vartika0611" target="_blank">GitHub</a> | 
<a href="https://www.linkedin.com/in/vartika-singh-56b95829b/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
