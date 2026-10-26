"""
Multimodal UI Components for TRACE-Tutor Dashboard

Provides file upload, audio recording, and rich media display widgets.
"""

import streamlit as st
from pathlib import Path
import tempfile
from typing import Optional, Dict
import base64


def render_multimodal_input() -> Dict:
    """
    Render multimodal input widgets.
    
    Returns:
        Dict with keys: text_query, image_path, audio_path
    """
    result = {
        "text_query": None,
        "image_path": None,
        "audio_path": None
    }
    
    # Tabs for different input modalities
    tab1, tab2, tab3 = st.tabs(["📝 Text", "🖼️ Image", "🎤 Audio"])
    
    with tab1:
        st.markdown("#### Type your question")
        text_input = st.text_area(
            "Question",
            height=120,
            placeholder="e.g., What is photosynthesis?",
            label_visibility="collapsed"
        )
        result["text_query"] = text_input
    
    with tab2:
        st.markdown("#### Upload an image or diagram")
        st.caption("📸 Upload a photo of a diagram, textbook page, or homework problem")
        
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed"
        )
        
        if uploaded_image:
            # Save to temp file
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_image.name).suffix)
            temp_image.write(uploaded_image.read())
            temp_image.close()
            
            result["image_path"] = temp_image.name
            
            # Display preview
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            # Text query for image context
            st.markdown("**Optional: Add a question about this image**")
            image_text = st.text_input(
                "Question about image",
                placeholder="e.g., Explain what this diagram shows",
                label_visibility="collapsed"
            )
            if image_text:
                result["text_query"] = image_text
    
    with tab3:
        st.markdown("#### Record or upload audio")
        st.caption("🎙️ Ask your question by voice (English, Hindi, or Urdu)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Upload audio file**")
            uploaded_audio = st.file_uploader(
                "Choose audio",
                type=["wav", "mp3", "m4a", "ogg"],
                label_visibility="collapsed"
            )
            
            if uploaded_audio:
                # Save to temp file
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_audio.name).suffix)
                temp_audio.write(uploaded_audio.read())
                temp_audio.close()
                
                result["audio_path"] = temp_audio.name
                
                # Play audio
                st.audio(uploaded_audio, format=f"audio/{Path(uploaded_audio.name).suffix[1:]}")
        
        with col2:
            st.markdown("**Record from microphone**")
            st.info("🚧 Coming soon: Browser-based recording")
            st.caption("For now, please upload a pre-recorded audio file")
    
    return result


def render_visual_evidence(evidence_list, clip_encoder=None):
    """
    Render visual evidence (images, diagrams) in a grid.
    
    Args:
        evidence_list: List of Evidence objects with figure_path
        clip_encoder: Optional CLIP encoder for similarity display
    """
    st.markdown("### 🖼️ Visual Evidence")
    
    visual_items = [e for e in evidence_list if e.figure_path and Path(e.figure_path).exists()]
    
    if not visual_items:
        st.info("No visual evidence found for this question")
        return
    
    # Display in grid (3 columns)
    cols = st.columns(3)
    
    for idx, evidence in enumerate(visual_items):
        with cols[idx % 3]:
            try:
                st.image(
                    evidence.figure_path,
                    caption=evidence.figure_caption or f"Source: Page {evidence.page}",
                    use_column_width=True
                )
                
                # Show metadata
                with st.expander("📋 Details"):
                    st.markdown(f"**Chapter:** {evidence.chapter_id}")
                    st.markdown(f"**Page:** {evidence.page}")
                    if evidence.figure_caption:
                        st.markdown(f"**Caption:** {evidence.figure_caption}")
                    if evidence.alt_text:
                        st.markdown(f"**Description:** {evidence.alt_text}")
            except Exception as e:
                st.error(f"Failed to load image: {e}")


def render_audio_player(audio_path: str, label: str = "Listen to Answer"):
    """
    Render audio player for TTS output.
    
    Args:
        audio_path: Path to audio file
        label: Label for audio player
    """
    if not audio_path or not Path(audio_path).exists():
        return
    
    st.markdown(f"### 🔊 {label}")
    
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error(f"Failed to play audio: {e}")


def render_tts_controls(answer_text: str, grade: int, language: str = "en"):
    """
    Render text-to-speech controls.
    
    Args:
        answer_text: Answer text to synthesize
        grade: Grade level
        language: Language code
    
    Returns:
        Path to generated audio file if button clicked, else None
    """
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 🔊 Listen to Answer")
    
    with col2:
        tts_enabled = st.checkbox("Enable audio", value=False, key="tts_toggle")
    
    with col3:
        slow_speech = st.checkbox("Slow speech", value=(grade <= 2), key="tts_slow")
    
    if tts_enabled:
        if st.button("🎙️ Generate Audio", type="primary"):
            with st.spinner("Generating audio..."):
                try:
                    from trace_tutor.audio_processor import create_audio_response
                    audio_path = create_audio_response(answer_text, grade=grade, language=language)
                    
                    if audio_path:
                        st.session_state.current_audio = audio_path
                        render_audio_player(audio_path, "Your Answer")
                        st.success("✅ Audio ready!")
                        return audio_path
                    else:
                        st.error("Failed to generate audio")
                except Exception as e:
                    st.error(f"TTS error: {e}")
        
        # Display previously generated audio
        if hasattr(st.session_state, 'current_audio') and st.session_state.current_audio:
            render_audio_player(st.session_state.current_audio, "Your Answer")
    
    return None


def render_multimodal_answer_card(answer, show_visual: bool = True, show_audio: bool = True):
    """
    Enhanced answer card with multimodal evidence display.
    
    Args:
        answer: Answer object
        show_visual: Whether to show visual evidence
        show_audio: Whether to show audio controls
    """
    st.markdown("## 💡 Answer")
    
    # Main answer text
    st.markdown(f"<div class='answer-box'>{answer.answer}</div>", unsafe_allow_html=True)
    
    # Visual evidence
    if show_visual and answer.visual_evidence_paths:
        st.markdown("---")
        st.markdown("### 📸 Visual Evidence")
        
        cols = st.columns(min(3, len(answer.visual_evidence_paths)))
        for idx, img_path in enumerate(answer.visual_evidence_paths):
            with cols[idx % len(cols)]:
                if Path(img_path).exists():
                    st.image(img_path, use_column_width=True)
                    st.caption(f"Evidence {idx + 1}")
    
    # Explanation
    if answer.explanation:
        with st.expander("📖 Detailed Explanation"):
            st.markdown(answer.explanation)
    
    # Next steps
    if answer.next_step:
        st.info(f"💡 **Next Step:** {answer.next_step}")
    
    # Audio output
    if show_audio:
        render_tts_controls(answer.answer, grade=3)  # Default grade 3
    
    # Source citations
    if answer.source_pages:
        st.markdown("---")
        st.markdown("### 📚 Sources")
        st.caption(f"Evidence from pages: {', '.join(map(str, sorted(set(answer.source_pages))))}")


def render_transcription_display(transcription: Dict):
    """
    Display audio transcription results.
    
    Args:
        transcription: Dict with text, language, confidence
    """
    if not transcription or not transcription.get("text"):
        return
    
    st.markdown("### 📝 Transcription")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.text_area(
            "Transcribed text",
            value=transcription["text"],
            height=80,
            label_visibility="collapsed"
        )
    
    with col2:
        lang_name = {
            "en": "🇬🇧 English",
            "hi": "🇮🇳 Hindi",
            "ur": "🇵🇰 Urdu"
        }.get(transcription.get("language", "unknown"), "Unknown")
        st.metric("Language", lang_name)
    
    with col3:
        confidence = transcription.get("confidence", 0.0)
        st.metric("Confidence", f"{confidence:.0%}")
    
    if confidence < 0.7:
        st.warning("⚠️ Low transcription confidence. Please verify the text is correct.")


def get_file_icon(modality: str) -> str:
    """Get emoji icon for modality type."""
    icons = {
        "text": "📝",
        "image": "🖼️",
        "table": "📊",
        "audio": "🎤",
        "video": "🎥"
    }
    return icons.get(modality, "📄")


def render_evidence_modality_badge(evidence):
    """Render modality badge for evidence."""
    icon = get_file_icon(evidence.modality)
    st.markdown(
        f'<span class="source-badge">{icon} {evidence.modality.title()}</span>',
        unsafe_allow_html=True
    )
