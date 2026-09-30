import time

import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; max-width: 820px; }
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #6d5dfc 0%, #b86bff 55%, #ff7eb3 100%);
        color: white;
    }
    .hero h1 { color: white; margin: 0 0 .3rem 0; font-size: 2.1rem; }
    .hero p { margin: 0; opacity: .92; font-size: 1rem; }
    .result {
        border-radius: 16px; padding: 1.3rem 1.5rem; margin: .6rem 0 1rem 0;
        display: flex; align-items: center; gap: 1.1rem;
    }
    .result.pos { background: rgba(33,195,84,.12); border: 1px solid rgba(33,195,84,.45); }
    .result.neg { background: rgba(255,75,75,.12); border: 1px solid rgba(255,75,75,.45); }
    .result .emoji { font-size: 3rem; line-height: 1; }
    .result .label { font-size: 1.6rem; font-weight: 700; letter-spacing: .5px; }
    .result .sub { opacity: .8; font-size: .92rem; }
    .bar-wrap { background: rgba(128,128,128,.18); border-radius: 999px; height: 12px; overflow: hidden; }
    .bar { height: 100%; border-radius: 999px; }
    .bar-row { display:flex; justify-content:space-between; font-size:.88rem; margin: .5rem 0 .25rem 0; }
    .tip { font-size: .85rem; opacity: .75; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model ----------
@st.cache_resource(show_spinner="Loading the AI model (first time only)...")
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


def analyze(text: str) -> dict:
    """Return label, confidence and both class scores for one text."""
    scores = load_model()(text, truncation=True, top_k=None)
    probs = {s["label"]: s["score"] for s in scores}
    label = max(probs, key=probs.get)
    return {
        "label": label,
        "confidence": probs[label],
        "positive": probs.get("POSITIVE", 0.0),
        "negative": probs.get("NEGATIVE", 0.0),
    }


def strength(conf: float) -> str:
    if conf >= 0.95:
        return "Very sure"
    if conf >= 0.80:
        return "Fairly sure"
    if conf >= 0.65:
        return "Leaning"
    return "Unsure (mixed or neutral text)"


def bar(label: str, value: float, color: str) -> str:
    return (
        f'<div class="bar-row"><span>{label}</span><span>{value:.1%}</span></div>'
        f'<div class="bar-wrap"><div class="bar" style="width:{value*100:.1f}%;background:{color}"></div></div>'
    )


# ---------- State ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "text" not in st.session_state:
    st.session_state.text = ""

EXAMPLES = {
    "😊 Happy review": "The food was amazing and the staff were so friendly!",
    "😡 Complaint": "My order arrived two weeks late and the box was broken. Never ordering again.",
    "🙃 Sarcasm": "Oh great, another Monday stuck in traffic for two hours. Just what I needed.",
    "😐 Mixed": "The phone looks beautiful but the battery dies before lunch.",
}

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <h1>💬 Sentiment Analyzer</h1>
      <p>Type any sentence and the AI tells you if it sounds positive or negative, and how sure it is.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_history, tab_about = st.tabs(
    ["✍️ Analyze", "📋 Batch", "📈 History", "ℹ️ About"]
)

# ---------- Single analysis ----------
with tab_single:
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (name, sample) in zip(cols, EXAMPLES.items()):
        if col.button(name, width="stretch"):
            st.session_state.text = sample

    text = st.text_area(
        "Your text",
        key="text",
        height=120,
        max_chars=2000,
        placeholder="e.g. I absolutely loved this product!",
        label_visibility="collapsed",
    )
    words = len(text.split())
    st.markdown(f'<div class="tip">{words} word{"s" if words != 1 else ""}</div>', unsafe_allow_html=True)

    go = st.button("Analyze sentiment ✨", type="primary", width="stretch")

    if go:
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Reading your text..."):
                r = analyze(text)
            st.session_state.history.append(
                {
                    "Time": time.strftime("%H:%M:%S"),
                    "Text": text,
                    "Label": r["label"],
                    "Confidence": round(r["confidence"], 4),
                    "Positive": r["positive"],
                }
            )

            pos = r["label"] == "POSITIVE"
            emoji = "😊" if pos else "😞"
            cls = "pos" if pos else "neg"
            st.markdown(
                f"""
                <div class="result {cls}">
                  <div class="emoji">{emoji}</div>
                  <div>
                    <div class="label">{r["label"]}</div>
                    <div class="sub">{strength(r["confidence"])} · {r["confidence"]:.1%} confidence</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                bar("😊 Positive", r["positive"], "#21c354")
                + bar("😞 Negative", r["negative"], "#ff4b4b"),
                unsafe_allow_html=True,
            )

            if r["confidence"] < 0.65:
                st.info("The model is unsure. This often happens with mixed or neutral sentences.")
            if pos and r["confidence"] > 0.9:
                st.caption("Heads up: this model can't detect sarcasm. Try the 🙃 Sarcasm example to see it fail.")

# ---------- Batch ----------
with tab_batch:
    st.write("Paste several sentences, **one per line**, to analyze them all at once.")
    batch = st.text_area(
        "Batch text",
        height=170,
        placeholder="Great service!\nThe app keeps crashing.\nDelivery was on time.",
        label_visibility="collapsed",
        key="batch",
    )
    if st.button("Analyze all 🚀", type="primary", width="stretch"):
        lines = [l.strip() for l in batch.splitlines() if l.strip()][:50]
        if not lines:
            st.warning("Please enter at least one line.")
        else:
            rows = []
            progress = st.progress(0.0, text="Analyzing...")
            for i, line in enumerate(lines, 1):
                r = analyze(line)
                rows.append(
                    {
                        "Text": line,
                        "Sentiment": ("😊 " if r["label"] == "POSITIVE" else "😞 ") + r["label"],
                        "Confidence": r["confidence"],
                    }
                )
                progress.progress(i / len(lines), text=f"Analyzed {i} of {len(lines)}")
            progress.empty()

            df = pd.DataFrame(rows)
            n_pos = df["Sentiment"].str.contains("POSITIVE").sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Sentences", len(df))
            c2.metric("😊 Positive", int(n_pos))
            c3.metric("😞 Negative", int(len(df) - n_pos))

            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )
            st.download_button(
                "⬇️ Download results (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )

# ---------- History ----------
with tab_history:
    hist = st.session_state.history
    if not hist:
        st.info("Your analyses will appear here. Go to ✍️ Analyze and try a few sentences.")
    else:
        df = pd.DataFrame(hist)
        n_pos = (df["Label"] == "POSITIVE").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Analyzed", len(df))
        c2.metric("😊 Positive", int(n_pos))
        c3.metric("Avg confidence", f"{df['Confidence'].mean():.1%}")

        st.caption("Positive score for each analysis (above 0.5 = positive)")
        st.line_chart(df.reset_index()[["index", "Positive"]].set_index("index"), height=200)

        st.dataframe(
            df[["Time", "Text", "Label", "Confidence"]].iloc[::-1],
            width="stretch",
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

# ---------- About ----------
with tab_about:
    st.markdown(
        """
        **How it works**
        This app uses **DistilBERT**, a small version of Google's BERT language model,
        fine-tuned on thousands of movie review sentences (the SST-2 dataset).
        It reads your text and outputs a probability for POSITIVE and NEGATIVE.

        **Known limits**
        - Only two classes, so neutral text is forced into one of them.
        - It can't reliably detect sarcasm.
        - Trained on English, so Roman Urdu or mixed-language text gives unreliable results.
        - Very long text is cut at about 512 tokens.

        **Built with** Streamlit and Hugging Face Transformers.
        """
    )
