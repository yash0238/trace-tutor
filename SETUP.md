# TRACE-Tutor Setup Guide

Complete step-by-step setup instructions for running TRACE-Tutor research system.

## Prerequisites

- **Python**: 3.10 or higher
- **RAM**: 16GB minimum (32GB recommended)
- **Disk**: 100GB free space
- **OS**: Windows, Linux, or macOS
- **Ollama**: For local LLM inference (optional but recommended)

## Step 1: Environment Setup

### Windows (PowerShell)

```powershell
# Navigate to project
cd trace-tutor

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Linux/macOS

```bash
cd trace-tutor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2: Install Ollama (Local LLM)

### Option A: Ollama (Recommended for CPU)

1. Download and install Ollama from https://ollama.ai
2. Pull a lightweight model:

```bash
# Phi-3 Mini (3B parameters, ~2GB)
ollama pull phi3:3b-mini-instruct

# Alternative: Llama 3.2 3B
# ollama pull llama3.2:3b-instruct

# Alternative: Qwen 2.5 3B
# ollama pull qwen2.5:3b-instruct
```

3. Test Ollama:

```bash
ollama list
```

### Option B: API-based (OpenAI/Anthropic)

If using API instead of local models:

```bash
# Set API key
export LLM_API_KEY="your-api-key-here"  # Linux/Mac
# or
$env:LLM_API_KEY="your-api-key-here"  # Windows PowerShell
```

Update `configs/study.yaml`:

```yaml
generation:
  provider: "openai"  # or "anthropic"
  api_key_env: "LLM_API_KEY"
  api_model: "gpt-3.5-turbo"  # or "claude-3-haiku-20240307"
```

## Step 3: Download NCERT Textbooks

```bash
# Download EVS textbooks for grades 3-5
python scripts/download_ncert.py --subject evs --grades 3 4 5

# Check downloads
ls data/raw/
```

**Expected output:**
- `ncert_evs_grade3.pdf`
- `ncert_evs_grade4.pdf`
- `ncert_evs_grade5.pdf`

**Note:** If downloads fail, manually download from https://ncert.nic.in/textbook.php and place in `data/raw/`.

## Step 4: Build Evidence Corpus

```bash
# Extract PDFs and create evidence bundles
python scripts/build_corpus.py --config configs/study.yaml

# This will:
# - Extract text and figures from PDFs
# - Create evidence bundles with role labels
# - Generate prerequisite graph
# - Save to data/bundles/
```

**Expected output:**
- `data/bundles/corpus_all.jsonl` (~1000-3000 bundles)
- `data/graph/prerequisites_sample.jsonl`
- `data/bundles/corpus_stats.json`

## Step 5: Build Retrieval Indices

```bash
# Pre-compute embeddings and build indices
python scripts/build_index.py --config configs/study.yaml

# This may take 5-15 minutes on CPU
```

**Expected output:**
- `data/interim/retriever_index.pkl`
- `data/interim/cache/embeddings_*.pkl`

## Step 6: Run Test Query

```python
# Test the system
python -c "
from trace_tutor import TRACETutor

tutor = TRACETutor(
    config_path='configs/study.yaml',
    bundles_path='data/bundles/corpus_all.jsonl'
)

response = tutor.ask(
    question='Why do plants need sunlight?',
    grade=4,
    subject='EVS'
)

print('Answer:', response.answer)
print('Sources:', response.source_pages)
print('Time:', f'{response.total_time:.2f}s')
"
```

## Step 7: Run Full Experiment

```bash
# Run experiment with example questions
python scripts/run_experiment.py \
  --config configs/study.yaml \
  --gold-questions data/gold/example_questions.jsonl \
  --bundles data/bundles/corpus_all.jsonl

# Results saved to runs/study_TIMESTAMP/
```

## Step 8: Create Gold Questions (For Research)

Create your own gold questions file at `data/gold/questions_pilot.jsonl`:

```jsonl
{"id": "q001", "text": "Your question here?", "grade": 4, "subject": "EVS", "query_type": "conceptual", "target_lo": "your_lo_id", "needs_visual": false, "mastery": {}}
```

See `data/gold/example_questions.jsonl` for more examples.

## Common Issues & Solutions

### Issue: Ollama not found

**Solution:**
```bash
# Check if Ollama is installed
ollama --version

# If not, install from https://ollama.ai
# Then pull a model:
ollama pull phi3:3b-mini-instruct
```

### Issue: Out of memory during embedding

**Solution:**
Reduce batch size in `configs/study.yaml`:

```yaml
retrieval:
  dense:
    batch_size: 16  # Reduce from 32
```

### Issue: PDF extraction fails

**Solution:**
```bash
# Install system dependencies (Linux)
sudo apt-get install tesseract-ocr

# Or download manually from NCERT and place in data/raw/
```

### Issue: Slow inference

**Solutions:**
1. Use smaller model: `phi3:3b-mini-instruct` instead of `phi3:medium`
2. Reduce max_tokens in config
3. Use API provider (OpenAI/Anthropic) if available

## Verification Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Ollama installed and model pulled (or API key set)
- [ ] NCERT PDFs downloaded (3 files in `data/raw/`)
- [ ] Corpus built (`data/bundles/corpus_all.jsonl` exists)
- [ ] Index built (`data/interim/retriever_index.pkl` exists)
- [ ] Test query runs successfully
- [ ] Example experiment completes

## Next Steps

1. **Manual Bundle Review**: Check `data/bundles/` and refine role labels
2. **Prerequisite Graph**: Review and validate `data/graph/prerequisites_sample.jsonl`
3. **Create Gold Questions**: Develop 200-400 annotated questions
4. **Run Ablations**: Compare different configurations
5. **Human Evaluation**: Get teacher/expert ratings

## File Structure Check

Your directory should look like this:

```
trace-tutor/
├── README.md
├── SETUP.md (this file)
├── configs/
│   └── study.yaml
├── data/
│   ├── raw/                     # Downloaded PDFs
│   ├── bundles/                 # Evidence bundles
│   ├── graph/                   # Prerequisite graph
│   ├── gold/                    # Gold questions
│   └── interim/                 # Indices and cache
├── src/trace_tutor/             # Source code
├── scripts/                     # Utility scripts
├── runs/                        # Experiment results
└── .venv/                       # Virtual environment
```

## Getting Help

1. **Documentation**: See `README.md` for project overview
2. **Configuration**: Check `configs/study.yaml` for all settings
3. **Data Rights**: Review `docs/data_rights.md` for licensing
4. **Logs**: Check console output for detailed error messages

## Performance Benchmarks

On Intel i5 12th Gen (your hardware):

- **PDF Extraction**: ~1-2 minutes per textbook
- **Embedding Generation**: ~5-10 minutes for 2000 bundles (CPU)
- **Single Query**: ~2-5 seconds (with local Phi-3 3B)
- **Full Experiment** (300 questions): ~20-40 minutes

## Optional: Development Tools

```bash
# Install development dependencies
pip install pytest pytest-cov black isort

# Run tests
pytest tests/

# Format code
black src/ scripts/
isort src/ scripts/
```

---

**Setup Complete!** 🎉

You're now ready to run TRACE-Tutor experiments. For research workflows, see the execution plan in the research documents.
