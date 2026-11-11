import streamlit as st
import time
from toolkit.summarizer import TextSummarizer
from toolkit.classifier import TextClassifier
from toolkit.extractor import InfoExtractor
from toolkit.rag import SimpleRAGPipeline

st.set_page_config(
    page_title="AI Text Processing Toolkit",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Text Processing Toolkit")
st.markdown("Interactive UI for Summarization, Zero-Shot Classification, Information Extraction, and RAG.")

# Cache model instances so they don't reload on every UI interaction
@st.cache_resource
def load_summarizer():
    return TextSummarizer()

@st.cache_resource
def load_classifier():
    return TextClassifier()

@st.cache_resource
def load_extractor():
    return InfoExtractor()

@st.cache_resource
def load_rag():
    return SimpleRAGPipeline()

# UI Tabs for each module
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Summarization", 
    "🏷️ Classification", 
    "🔍 Information Extraction", 
    "📚 Document RAG"
])

# ---------------------------------------------------------
# Tab 1: Summarization
# ---------------------------------------------------------
with tab1:
    st.subheader("Abstractive Text Summarization")
    input_text = st.text_area(
        "Enter text to summarize:",
        value="Artificial intelligence is transforming industries worldwide, from healthcare and finance to education and transportation. Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language in valuable ways. Modern Transformer architectures have drastically improved long-term semantic context handling across documents.",
        height=150
    )
    col1, col2 = st.columns(2)
    with col1:
        min_len = st.slider("Min Length (tokens)", 10, 100, 30)
    with col2:
        max_len = st.slider("Max Length (tokens)", 50, 250, 130)

    if st.button("Generate Summary", key="btn_summary"):
        with st.spinner("Processing with Seq2Seq model..."):
            summarizer = load_summarizer()
            res = summarizer.summarize(input_text, min_length=min_len, max_length=max_len)
            
            st.success("Summary Generated!")
            st.markdown(f"**Summary:** {res.data['summary_text']}")
            
            # Metric badges
            m1, m2, m3 = st.columns(3)
            m1.metric("Latency", f"{res.metadata.latency_ms} ms")
            m2.metric("Compression Ratio", f"{res.data['compression_ratio']}")
            m3.metric("Input Tokens", f"{res.metadata.input_token_count}")
            
            with st.expander("View Raw JSON Output"):
                st.json(res.model_dump())

# ---------------------------------------------------------
# Tab 2: Classification
# ---------------------------------------------------------
with tab2:
    st.subheader("Zero-Shot Classification")
    clf_text = st.text_area(
        "Enter text to classify:",
        value="I was billed twice on my credit card for the monthly subscription. Can you reverse the charge?",
        height=100
    )
    labels_input = st.text_input(
        "Enter candidate labels (comma-separated):",
        value="billing issue, technical support, sales inquiry, account cancellation"
    )
    multi_label = st.checkbox("Multi-label classification (independent probabilities)", value=False)

    if st.button("Classify Text", key="btn_classify"):
        labels = [l.strip() for l in labels_input.split(",") if l.strip()]
        if not labels:
            st.error("Please provide at least one candidate label.")
        else:
            with st.spinner("Classifying text..."):
                classifier = load_classifier()
                res = classifier.classify(clf_text, candidate_labels=labels, multi_label=multi_label)
                
                st.success(f"Predicted Top Category: **{res.data['top_label']}** (Confidence: {res.data['top_score']:.2%})")
                st.metric("Latency", f"{res.metadata.latency_ms} ms")
                
                # Bar chart of scores
                scores_dict = {item["label"]: item["score"] for item in res.data["all_scores"]}
                st.bar_chart(scores_dict)

                with st.expander("View Raw JSON Output"):
                    st.json(res.model_dump())

# ---------------------------------------------------------
# Tab 3: Information Extraction
# ---------------------------------------------------------
with tab3:
    st.subheader("Named Entity Recognition (NER)")
    ner_text = st.text_area(
        "Enter text to extract entities from:",
        value="Satya Nadella, CEO of Microsoft, announced an investment of $10 billion in OpenAI in Seattle, Washington.",
        height=100
    )

    if st.button("Extract Entities", key="btn_ner"):
        with st.spinner("Extracting entities..."):
            extractor = load_extractor()
            res = extractor.extract_entities(ner_text)
            
            st.success("Entities Extracted!")
            st.metric("Latency", f"{res.metadata.latency_ms} ms")
            
            # Display grouped entities
            st.markdown("#### Categorized Entities")
            for cat, items in res.data["grouped_entities"].items():
                st.markdown(f"- **{cat}:** {', '.join(items)}")

            with st.expander("View Structured Entity Spans"):
                st.table(res.data["entities"])

            with st.expander("View Raw JSON Output"):
                st.json(res.model_dump())

# ---------------------------------------------------------
# Tab 4: Local RAG
# ---------------------------------------------------------
with tab4:
    st.subheader("Retrieval-Augmented Generation (Local RAG)")
    rag_doc = st.text_area(
        "Knowledge Document Context:",
        value="The Apollo space program was conducted by NASA between 1961 and 1972. Its goal was to land the first humans on the Moon. Apollo 11 succeeded in this goal on July 20, 1969, when astronauts Neil Armstrong and Buzz Aldrin landed the Lunar Module Eagle. Neil Armstrong was the first person to walk on the lunar surface, followed nineteen minutes later by Aldrin. The program used the Saturn V rocket.",
        height=120
    )
    rag_question = st.text_input("Question:", value="Who was the second person to walk on the Moon?")
    top_k = st.slider("Context chunks to retrieve (Top K):", 1, 3, 2)

    if st.button("Ask Question", key="btn_rag"):
        with st.spinner("Retrieving semantic chunks and generating answer..."):
            rag = load_rag()
            rag.index_documents([rag_doc], chunk_size=40)
            res = rag.query(rag_question, top_k=top_k)
            
            st.success("Answer Generated!")
            st.markdown(f"**Answer:** {res.data['answer']}")
            st.metric("Latency", f"{res.metadata.latency_ms} ms")
            
            st.markdown("#### Retrieved Supporting Context Chunks")
            for chunk in res.data["context_chunks"]:
                st.info(f"Chunk #{chunk['chunk_id']} (Similarity: {chunk['similarity_score']}):\n{chunk['content']}")

            with st.expander("View Raw JSON Output"):
                st.json(res.model_dump())