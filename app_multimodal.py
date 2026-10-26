"""
TRACE-Tutor Complete Multimodal Dashboard

Production-ready app with all features:
- Image upload and understanding
- Audio input (speech-to-text)
- Audio output (text-to-speech)
- Cross-modal retrieval
- Faithfulness gate
- Clarification agent
- Advanced retrieval components
"""

import streamlit as st
import sys
from pathlib import Path
import time
import tempfile
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trace_tutor.tutor import TRACETutor
from trace_tutor.schema import Query

# Page configuration
st.set_page_config(
    page_title="TRACE-Tutor Multimodal | AI Learning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    .evidence-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .visual-evidence-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_tutor():
    """Load and cache TRACE-Tutor instance."""
    return TRACETutor(
        config_path="configs/study.yaml",
        bundles_path="data/bundles/corpus_all.jsonl",
        graph_path="data/graph/prerequisites.jsonl"
    )


def render_header():
    """Render header."""
    st.markdown('<h1 class="main-header">🎓 TRACE-Tutor Multimodal</h1>', unsafe_allow_html=True)
    st.markdown("**Complete AI Learning Assistant** | Text • Image • Audio")
    st.markdown("---")


def render_sidebar(tutor):
    """Render configuration sidebar."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        grade = st.selectbox("Grade Level", [1, 2, 3, 4], index=2, key="grade_select")
        
        subject = st.selectbox(
            "Subject",
            ["Science", "Math", "Environmental Studies", "Social Studies"],
            key="subject_select"
        )
        
        query_type = st.selectbox(
            "Question Type",
            ["conceptual", "direct_fact", "diagram", "worked_problem"],
            key="query_type_select"
        )
        
        st.markdown("---")
        st.markdown("### 🔧 Advanced Options")
        
        use_cache = st.checkbox("Use semantic cache", value=True, help="10-20x faster for similar questions")
        use_faithfulness = st.checkbox("Faithfulness gate", value=True, help="Verify answer support")
        use_clarification = st.checkbox("Clarification agent", value=True, help="Ask follow-ups for vague queries")
        show_evidence = st.checkbox("Show evidence", value=True)
        show_visual = st.checkbox("Show visual evidence", value=True)
        enable_tts = st.checkbox("Enable text-to-speech", value=False)
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        
        if hasattr(tutor, 'semantic_cache'):
            cache_stats = tutor.semantic_cache.get_stats()
            st.metric("Cache Hits", cache_stats['hits'])
            st.metric("Cache Size", cache_stats['size'])
            if cache_stats['hits'] + cache_stats['misses'] > 0:
                hit_rate = cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses'])
                st.metric("Hit Rate", f"{hit_rate:.1%}")
        
        return {
            "grade": grade,
            "subject": subject,
            "query_type": query_type,
            "use_cache": use_cache,
            "use_faithfulness": use_faithfulness,
            "use_clarification": use_clarification,
            "show_evidence": show_evidence,
            "show_visual": show_visual,
            "enable_tts": enable_tts
        }


def render_multimodal_input():
    """Render multimodal input tabs."""
    st.markdown("### 💬 Ask Your Question")
    
    tab1, tab2, tab3 = st.tabs(["📝 Text", "🖼️ Image", "🎤 Audio"])
    
    result = {
        "text_query": None,
        "image_path": None,
        "audio_path": None
    }
    
    with tab1:
        text_query = st.text_area(
            "Type your question",
            height=120,
            placeholder="e.g., How does photosynthesis work?",
            key="text_input"
        )
        result["text_query"] = text_query
    
    with tab2:
        st.markdown("#### Upload diagram or image")
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "webp"],
            key="image_upload"
        )
        
        if uploaded_image:
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_image.name).suffix)
            temp_img.write(uploaded_image.read())
            temp_img.close()
            result["image_path"] = temp_img.name
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            with col2:
                image_question = st.text_input(
                    "Question about this image",
                    placeholder="What does this diagram show?",
                    key="image_text"
                )
                if image_question:
                    result["text_query"] = image_question
    
    with tab3:
        st.markdown("#### Upload audio question")
        uploaded_audio = st.file_uploader(
            "Choose audio file",
            type=["wav", "mp3", "m4a", "ogg"],
            key="audio_upload"
        )
        
        if uploaded_audio:
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_audio.name).suffix)
            temp_audio.write(uploaded_audio.read())
            temp_audio.close()
            result["audio_path"] = temp_audio.name
            
            st.audio(uploaded_audio)
            st.info("Audio will be transcribed when you click 'Get Answer'")
    
    return result


def render_answer_card(answer, config):
    """Render answer with all features."""
    st.markdown("## 💡 Answer")
    
    # Status indicator
    if answer.cached:
        st.success("⚡ Cached response (10-20x faster!)")
    
    if answer.status == "clarify":
        st.warning(f"❓ **Clarification Needed**\n\n{answer.answer}")
        return
    
    if answer.status == "fallback":
        st.warning(f"⚠️ **Fallback Response**\n\n{answer.answer}")
        if answer.fallback_reason:
            st.caption(f"Reason: {answer.fallback_reason}")
        return
    
    # Main answer
    st.markdown(f"### Answer:\n{answer.answer}")
    
    # Faithfulness indicator
    if answer.verified:
        st.success(f"✅ Verified (Support: {answer.support_score:.0%})")
    else:
        st.warning(f"⚠️ Low confidence (Support: {answer.support_score:.0%})")
    
    # Visual evidence
    if config["show_visual"] and answer.visual_evidence_paths:
        st.markdown("---")
        st.markdown("### 📸 Visual Evidence")
        
        cols = st.columns(min(3, len(answer.visual_evidence_paths)))
        for idx, img_path in enumerate(answer.visual_evidence_paths):
            with cols[idx % len(cols)]:
                if Path(img_path).exists():
                    st.image(img_path, use_column_width=True)
                    st.caption(f"Evidence {idx + 1}")
    
    # Text evidence
    if config["show_evidence"]:
        with st.expander("📚 Evidence Sources"):
            st.markdown(f"**Evidence Bundles:** {len(answer.evidence_bundle_ids)}")
            st.markdown(f"**Source Pages:** {', '.join(map(str, sorted(set(answer.source_pages))))}")
            
            if answer.unsupported_claims:
                st.warning(f"**Unsupported Claims:** {len(answer.unsupported_claims)}")
                for claim in answer.unsupported_claims[:3]:
                    st.markdown(f"- {claim}")
    
    # TTS controls
    if config["enable_tts"]:
        st.markdown("---")
        st.markdown("### 🔊 Listen to Answer")
        
        if st.button("🎙️ Generate Audio", key="tts_btn"):
            with st.spinner("Generating audio..."):
                try:
                    tutor = load_tutor()
                    audio_path = tutor.generate_audio_response(
                        answer.answer,
                        grade=config["grade"],
                        language="en"
                    )
                    
                    if audio_path and Path(audio_path).exists():
                        with open(audio_path, "rb") as f:
                            st.audio(f.read(), format="audio/mp3")
                        st.success("✅ Audio ready!")
                    else:
                        st.error("Failed to generate audio")
                except Exception as e:
                    st.error(f"TTS error: {e}")
    
    # Performance metrics
    with st.expander("⚡ Performance"):
        cols = st.columns(4)
        cols[0].metric("Total Time", f"{answer.total_time:.2f}s")
        cols[1].metric("Generation", f"{answer.generation_time:.2f}s")
        cols[2].metric("Retrieval", f"{answer.retrieval_time:.2f}s")
        cols[3].metric("Selection", f"{answer.selection_time:.2f}s")


def main():
    """Main application."""
    render_header()
    
    # Initialize
    try:
        tutor = load_tutor()
    except Exception as e:
        st.error(f"❌ Failed to load tutor: {e}")
        st.stop()
    
    # Sidebar config
    config = render_sidebar(tutor)
    
    # Main area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Multimodal input
        inputs = render_multimodal_input()
        
        # Submit button
        if st.button("🎯 Get Answer", type="primary", use_container_width=True):
            text_query = inputs["text_query"]
            
            if not text_query or len(text_query.strip()) < 3:
                st.warning("⚠️ Please enter a valid question (at least 3 characters)")
                st.stop()
            
            # Process query
            with st.spinner("🤔 Processing... (retrieving evidence, generating answer)"):
                try:
                    answer = tutor.ask_multimodal(
                        question=text_query,
                        grade=config["grade"],
                        subject=config["subject"],
                        image_path=inputs["image_path"],
                        audio_path=inputs["audio_path"],
                        query_type=config["query_type"],
                        use_cache=config["use_cache"],
                        use_faithfulness_gate=config["use_faithfulness"]
                    )
                    
                    # Display answer
                    render_answer_card(answer, config)
                    
                    # Feedback
                    st.markdown("---")
                    st.markdown("### 📊 Was this helpful?")
                    cols = st.columns(4)
                    if cols[0].button("👍 Very Helpful", use_container_width=True):
                        st.success("Thank you for your feedback!")
                    if cols[1].button("🙂 Helpful", use_container_width=True):
                        st.success("Thanks!")
                    if cols[2].button("😐 Somewhat", use_container_width=True):
                        st.info("We'll try to improve")
                    if cols[3].button("👎 Not Helpful", use_container_width=True):
                        st.warning("Sorry, we'll do better")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("🔍 Debug Info"):
                        import traceback
                        st.code(traceback.format_exc())
    
    with col_right:
        # Quick examples
        st.markdown("### ⚡ Quick Examples")
        
        examples = {
            "Science": [
                "How does photosynthesis work?",
                "What is the water cycle?",
                "Explain plant cell parts"
            ],
            "Math": [
                "How do I add fractions?",
                "What is multiplication?",
                "Explain place value"
            ]
        }
        
        subject_examples = examples.get(config["subject"], examples["Science"])
        
        for ex in subject_examples:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                st.session_state.text_input = ex
                st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ Features")
        st.markdown("""
        - ✅ **Multimodal Input**: Text, image, audio
        - ✅ **Cross-modal Search**: Find visual + text evidence
        - ✅ **Semantic Caching**: 10-20x faster
        - ✅ **Faithfulness Gate**: Verified answers
        - ✅ **Clarification Agent**: Handles vague queries
        - ✅ **Text-to-Speech**: Audio answers
        """)


if __name__ == "__main__":
    main()
