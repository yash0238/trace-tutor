# Quick Start: Multimodal TRACE-Tutor

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd c:\Users\DELL\Downloads\Ankita\trace-tutor

# Install all dependencies
pip install -r requirements.txt

# This includes:
# - openai-whisper (speech-to-text)
# - gtts (text-to-speech)
# - sentence-transformers (for CLIP)
# - google-generativeai (for image captioning)
# - pydub, opencv-python (media processing)
```

### Step 2: Set API Keys (Optional)

```bash
# For image captioning with Gemini Vision
# (Has fallback if not set)
export GEMINI_API_KEY=your_gemini_api_key_here

# Or create a .env file:
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Step 3: Test Installation

```bash
# Run the demo suite
python examples/multimodal_demo.py

# Select demo to run:
# 1. Image Encoding
# 2. Image Captioning  
# 3. Audio Transcription
# 4. Text-to-Speech
# 5. Multimodal Query
# 6. Visual Evidence
```

### Step 4: Try Multimodal Features

#### Option A: Python API

```python
from trace_tutor.tutor import TRACETutor
from trace_tutor.schema import Query
from trace_tutor.audio_processor import SpeechToText, create_audio_response

# Initialize tutor
tutor = TRACETutor(config_path="configs/study.yaml")

# Example 1: Text + Image Query
query = Query(
    id="q001",
    text="What is this diagram about?",
    image_path="path/to/diagram.png",
    grade=3,
    subject="Science"
)

response = tutor.ask(
    question=query.text,
    grade=query.grade,
    subject=query.subject
)

print(response.answer)

# Example 2: Audio Question
stt = SpeechToText()
transcription = stt.transcribe("question.mp3")

response = tutor.ask(
    question=transcription["text"],
    grade=2,
    subject="Math"
)

# Generate audio answer
audio_path = create_audio_response(response.answer, grade=2, language="en")
print(f"Audio saved to: {audio_path}")
```

#### Option B: Dashboard (Streamlit)

```bash
# Start the dashboard
streamlit run app.py

# Then:
# 1. Select Grade and Subject
# 2. Click "🖼️ Image" tab to upload diagrams
# 3. Click "🎤 Audio" tab to upload voice questions
# 4. After getting answer, enable "Read aloud" for TTS
```

---

## 📖 Common Use Cases

### Use Case 1: Student Uploads Homework Photo

```python
from trace_tutor.image_understanding import CLIPImageEncoder, ImageCaptionGenerator

# Generate caption for uploaded image
captioner = ImageCaptionGenerator()
caption = captioner.generate_caption("homework_photo.jpg")

# Or use CLIP for retrieval
encoder = CLIPImageEncoder()
similarity = encoder.compute_similarity(
    "quadratic equation solution",
    "homework_photo.jpg"
)

print(f"Relevance: {similarity:.2%}")
```

### Use Case 2: Voice Question in Hindi

```python
from trace_tutor.audio_processor import SpeechToText

stt = SpeechToText(model_size="base")
result = stt.transcribe("hindi_question.mp3", language="hi")

print(f"Text: {result['text']}")
print(f"Language: {result['language']}")
print(f"Confidence: {result['confidence']:.0%}")
```

### Use Case 3: Generate Audio Answer for Young Student

```python
from trace_tutor.audio_processor import create_audio_response

answer = "Plants make food using sunlight, water, and air. This is called photosynthesis."

# Slow speech for Grade 1
audio_path = create_audio_response(
    answer_text=answer,
    grade=1,  # Slower pace
    language="en"
)

# Play with any audio player
```

### Use Case 4: Cross-Modal Retrieval

```python
from trace_tutor.multimodal_retriever import MultimodalRetriever
from trace_tutor.schema import Query

retriever = MultimodalRetriever(
    text_retriever=hybrid_retriever,
    modality_weights={"text": 0.5, "image": 0.3, "audio": 0.2}
)

# Query with both text and image
query = Query(
    id="q001",
    text="Explain this diagram",
    image_path="cell_diagram.png",
    grade=4,
    subject="Science"
)

results = retriever.retrieve(query, k=10)

for evidence, score in results[:3]:
    print(f"Score: {score:.3f}")
    print(f"Text: {evidence.text[:100]}...")
    if evidence.figure_path:
        print(f"Image: {evidence.figure_path}")
```

---

## 🎨 Dashboard Integration Example

### Add Multimodal Input to app.py

```python
from trace_tutor.multimodal_ui import (
    render_multimodal_input,
    render_visual_evidence,
    render_tts_controls
)

# In your main app function:
def main():
    # Get multimodal inputs
    inputs = render_multimodal_input()
    
    text_query = inputs["text_query"]
    image_path = inputs["image_path"]
    audio_path = inputs["audio_path"]
    
    # Process audio if provided
    if audio_path:
        from trace_tutor.audio_processor import SpeechToText
        stt = SpeechToText()
        transcription = stt.transcribe(audio_path)
        text_query = transcription["text"]
        st.info(f"🎤 Detected: {transcription['language']}")
    
    # Get answer
    if st.button("Get Answer"):
        response = tutor.ask(
            question=text_query,
            grade=grade,
            subject=subject
        )
        
        # Display answer
        st.markdown(response.answer)
        
        # Show visual evidence if available
        if response.visual_evidence_paths:
            render_visual_evidence(response.visual_evidence_paths)
        
        # Add TTS controls
        render_tts_controls(response.answer, grade=grade)
```

---

## 📊 Performance Tips

### 1. Cache CLIP Embeddings

```python
# First run: slow (downloads model)
encoder = CLIPImageEncoder()
embedding = encoder.encode_image("diagram.png")  # ~2-3 seconds

# Subsequent runs: fast (model cached)
embedding = encoder.encode_image("diagram2.png")  # ~100ms
```

### 2. Use Smaller Whisper Models for Speed

```python
# Faster but less accurate
stt = SpeechToText(model_size="tiny")  # ~2-3x faster

# Balanced (recommended)
stt = SpeechToText(model_size="base")  # Good accuracy

# More accurate but slower
stt = SpeechToText(model_size="medium")  # ~3-4x slower
```

### 3. Batch Process Images

```python
encoder = CLIPImageEncoder()

# Process multiple images at once
image_paths = ["img1.png", "img2.png", "img3.png"]
embeddings = [encoder.encode_image(path) for path in image_paths]
```

---

## 🔧 Troubleshooting

### Problem: "No module named 'whisper'"
**Solution:**
```bash
pip install openai-whisper
```

### Problem: "CLIP model download is slow"
**Solution:**
```bash
# Download manually and cache
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')"
```

### Problem: "gTTS not working"
**Solution:**
```bash
pip install gtts pydub
# Also install ffmpeg for audio conversion
```

### Problem: "Image not loading in Streamlit"
**Solution:**
```python
# Use full absolute path
image_path = str(Path(image_path).resolve())
st.image(image_path)
```

### Problem: "Transcription is in wrong language"
**Solution:**
```python
# Specify language explicitly
result = stt.transcribe(audio_path, language="en")  # or "hi", "ur"
```

---

## 📈 What to Test First

### Priority 1: Image Upload & Display
1. Upload a diagram
2. See if it displays in the dashboard
3. Check if caption is generated

### Priority 2: Audio Transcription
1. Upload a voice question (MP3)
2. See if it transcribes correctly
3. Check language detection

### Priority 3: TTS Output
1. Get an answer from the system
2. Click "Enable audio"
3. Generate and play audio

### Priority 4: Cross-Modal Search
1. Upload image of photosynthesis diagram
2. Ask "How does this process work?"
3. Check if text + image evidence is retrieved

---

## 🎯 Example Workflow

### Full Multimodal Session

```python
# 1. Student uploads homework photo
from trace_tutor.image_understanding import ImageCaptionGenerator

captioner = ImageCaptionGenerator()
caption = captioner.generate_caption("homework.jpg")
print(f"Image is about: {caption}")

# 2. Student asks question via audio
from trace_tutor.audio_processor import SpeechToText

stt = SpeechToText()
transcription = stt.transcribe("question.mp3")
question = transcription["text"]
print(f"Question: {question}")

# 3. System retrieves multimodal evidence
from trace_tutor.tutor import TRACETutor

tutor = TRACETutor()
response = tutor.ask(question=question, grade=3, subject="Math")

# 4. Generate audio answer
from trace_tutor.audio_processor import create_audio_response

audio_path = create_audio_response(response.answer, grade=3)
print(f"Answer audio: {audio_path}")

# 5. Display visual evidence
for img_path in response.visual_evidence_paths:
    print(f"Evidence image: {img_path}")
```

---

## 📞 Need Help?

1. **Check logs**: Look for error messages in console
2. **Run demo**: `python examples/multimodal_demo.py`
3. **Read docs**: See `MULTIMODAL_IMPLEMENTATION.md`
4. **Check requirements**: Verify all packages installed

---

## ✅ Checklist

Before reporting issues, verify:
- [ ] All dependencies installed (`pip list | grep -E "whisper|gtts|sentence-transformers"`)
- [ ] GEMINI_API_KEY set (for captioning, optional)
- [ ] Test files exist (`data/interim/figures/` for images)
- [ ] Demo script runs (`python examples/multimodal_demo.py`)
- [ ] Dashboard starts (`streamlit run app.py`)

---

## 🎉 You're Ready!

Start with simple tests:
1. Upload an image → see if it displays
2. Upload audio → see if it transcribes
3. Generate TTS → see if audio plays

Then move to full workflow:
1. Multimodal query (text + image)
2. Cross-modal retrieval
3. Visual evidence display
4. Audio answer generation

**Have fun building multimodal learning experiences! 🚀**
