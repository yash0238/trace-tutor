"""
Multimodal TRACE-Tutor Demo

Demonstrates image input, audio processing, and cross-modal retrieval.
Run this to test the multimodal pipeline end-to-end.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trace_tutor.image_understanding import CLIPImageEncoder, ImageCaptionGenerator, create_visual_evidence
from trace_tutor.audio_processor import SpeechToText, TextToSpeech, create_audio_response
from trace_tutor.schema import Query, Evidence
import os


def demo_image_encoding():
    """Demo 1: Encode images with CLIP."""
    print("\n" + "="*60)
    print("DEMO 1: Image Encoding with CLIP")
    print("="*60)
    
    encoder = CLIPImageEncoder()
    
    # Example: encode a sample image (you need to provide an actual image path)
    sample_image = "data/interim/figures/sample_diagram.png"
    
    if not Path(sample_image).exists():
        print(f"⚠️  Image not found: {sample_image}")
        print("📁 Place a diagram in data/interim/figures/ to test")
        return
    
    print(f"\n📸 Encoding: {sample_image}")
    embedding = encoder.encode_image(sample_image)
    print(f"✅ Embedding shape: {embedding.shape}")
    print(f"✅ Dimension: {len(embedding)}")
    
    # Compute text-image similarity
    text_query = "photosynthesis diagram"
    similarity = encoder.compute_similarity(text_query, sample_image)
    print(f"\n🔍 Similarity between '{text_query}' and image: {similarity:.3f}")


def demo_image_captioning():
    """Demo 2: Generate captions with Gemini Vision."""
    print("\n" + "="*60)
    print("DEMO 2: Image Captioning (Gemini Vision)")
    print("="*60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set in environment")
        print("Set it with: export GEMINI_API_KEY=your_key_here")
        return
    
    captioner = ImageCaptionGenerator(api_key=api_key)
    
    sample_image = "data/interim/figures/sample_diagram.png"
    
    if not Path(sample_image).exists():
        print(f"⚠️  Image not found: {sample_image}")
        return
    
    print(f"\n📸 Generating caption for: {sample_image}")
    caption = captioner.generate_caption(sample_image, detail_level="detailed")
    print(f"\n📝 Caption: {caption}")


def demo_audio_transcription():
    """Demo 3: Transcribe audio with Whisper."""
    print("\n" + "="*60)
    print("DEMO 3: Audio Transcription (Whisper)")
    print("="*60)
    
    stt = SpeechToText(model_size="base")
    
    sample_audio = "data/test/sample_question.mp3"
    
    if not Path(sample_audio).exists():
        print(f"⚠️  Audio file not found: {sample_audio}")
        print("📁 Place an audio file in data/test/ to test")
        return
    
    print(f"\n🎤 Transcribing: {sample_audio}")
    result = stt.transcribe(sample_audio)
    
    print(f"\n✅ Transcription:")
    print(f"   Text: {result['text']}")
    print(f"   Language: {result['language']}")
    print(f"   Confidence: {result['confidence']:.1%}")


def demo_text_to_speech():
    """Demo 4: Generate audio from text."""
    print("\n" + "="*60)
    print("DEMO 4: Text-to-Speech (gTTS)")
    print("="*60)
    
    answer_text = (
        "Photosynthesis is the process by which plants make their own food. "
        "They use sunlight, water, and carbon dioxide to create glucose and oxygen. "
        "The green pigment chlorophyll helps capture sunlight energy."
    )
    
    print(f"\n📝 Input text ({len(answer_text)} chars):")
    print(f"   {answer_text[:100]}...")
    
    print("\n🔊 Generating audio...")
    audio_path = create_audio_response(answer_text, grade=3, language="en")
    
    if audio_path:
        print(f"✅ Audio saved to: {audio_path}")
        print(f"📊 File size: {Path(audio_path).stat().st_size / 1024:.1f} KB")
    else:
        print("❌ Audio generation failed")


def demo_multimodal_query():
    """Demo 5: Create multimodal query."""
    print("\n" + "="*60)
    print("DEMO 5: Multimodal Query Construction")
    print("="*60)
    
    # Example: Create a query with text + image
    query = Query(
        id="demo_q001",
        text="Explain what this diagram shows about plant cells",
        image_path="uploads/plant_cell_diagram.png",
        audio_transcription="Explain what this diagram shows",
        audio_language="en",
        audio_confidence=0.92,
        grade=4,
        subject="Science",
        query_type="diagram",
        needs_visual=True
    )
    
    print("\n📋 Query created:")
    print(f"   ID: {query.id}")
    print(f"   Text: {query.text}")
    print(f"   Image: {query.image_path}")
    print(f"   Audio transcription: {query.audio_transcription}")
    print(f"   Language: {query.audio_language}")
    print(f"   Grade: {query.grade}")
    print(f"   Subject: {query.subject}")
    
    print("\n✅ Schema validation passed!")


def demo_visual_evidence():
    """Demo 6: Create visual evidence bundle."""
    print("\n" + "="*60)
    print("DEMO 6: Visual Evidence Bundle")
    print("="*60)
    
    import numpy as np
    
    # Create dummy CLIP embedding
    dummy_embedding = np.random.randn(512)
    
    evidence = create_visual_evidence(
        image_path="data/interim/figures/photosynthesis_diagram.png",
        caption="Diagram showing the process of photosynthesis in plant leaves",
        grade=4,
        subject="Science",
        chapter="Ch5_Plant_Life",
        source_page=42,
        embedding=dummy_embedding
    )
    
    print("\n📦 Evidence bundle created:")
    print(f"   ID: {evidence.chunk_id}")
    print(f"   Text: {evidence.text[:80]}...")
    print(f"   Modality: {evidence.modality}")
    print(f"   Grade: {evidence.grade}")
    print(f"   Chapter: {evidence.chapter}")
    print(f"   Has embedding: {evidence.metadata.get('has_embedding')}")
    print(f"   Embedding dim: {evidence.metadata.get('embedding_dim')}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("🚀 TRACE-Tutor Multimodal Demo Suite")
    print("="*60)
    
    demos = [
        ("Image Encoding", demo_image_encoding),
        ("Image Captioning", demo_image_captioning),
        ("Audio Transcription", demo_audio_transcription),
        ("Text-to-Speech", demo_text_to_speech),
        ("Multimodal Query", demo_multimodal_query),
        ("Visual Evidence", demo_visual_evidence),
    ]
    
    print("\nAvailable demos:")
    for idx, (name, _) in enumerate(demos, 1):
        print(f"  {idx}. {name}")
    
    print("\nSelect demo to run (1-6, or 'all'):")
    choice = input("> ").strip().lower()
    
    if choice == "all":
        for name, func in demos:
            try:
                func()
            except Exception as e:
                print(f"\n❌ Demo '{name}' failed: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        idx = int(choice) - 1
        name, func = demos[idx]
        try:
            func()
        except Exception as e:
            print(f"\n❌ Demo '{name}' failed: {e}")
    else:
        print("Invalid choice. Exiting.")
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
