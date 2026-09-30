import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")


@st.cache_resource
def load():
    # Pinned model so it doesn't change silently and no warning is shown
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


st.title("Sentiment Analyzer")
st.caption("Type a sentence and get POSITIVE or NEGATIVE with a confidence score.")

txt = st.text_area("Your text:")

if st.button("Analyze"):
    if not txt.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            # truncation=True stops long paragraphs from crashing the model (512 token limit)
            r = load()(txt, truncation=True)[0]
        if r["label"] == "POSITIVE":
            st.success(r["label"])
        else:
            st.error(r["label"])
        st.write(f"Confidence: {r['score']:.1%}")
