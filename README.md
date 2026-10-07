# TRACE-Tutor: Trajectory- and Role-Aware Curriculum Evidence Tutor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A trustworthy, curriculum-grounded AI tutor for primary education that uses pedagogically-complete evidence selection instead of generic top-k RAG.

## 🎯 Research Question

**Does curriculum evidence bundling with role-constrained, prerequisite-licensed selection improve grounded pedagogical quality over conventional retrieval?**

## 🚀 Key Features

- **📚 Curriculum Evidence Bundles**: Source-linked units preserving text, diagrams, exercises, learning outcomes, and prerequisites
- **🎓 Prerequisite Licensing**: Lower-grade content admitted only when verified and needed
- **🔍 PS-Evidence Selection**: Minimal, role-complete evidence sets (max 3 bundles)
- **✅ TRACE Verification**: Answer claims checked against selected evidence before serving
- **♿ Accessibility**: Simplified text, alt-text, audio scripts
- **🔒 Privacy-by-Design**: Pseudonymization, minimization, audit logs
- **📊 Comprehensive Evaluation**: Retrieval, grounding, grade-fit, role coverage metrics

## 🏗️ Architecture

```
Query + Grade + Learner State
         ↓
    Query Interpreter
         ↓
    Curriculum Scope Filter
         ↓
    Prerequisite Gap Check
         ↓
BM25 + Dense + Visual Retrieval
         ↓
    RRF Fusion + Reranking
         ↓
    PS-Evidence Selector (max 3)
         ↓
    Grounded Answer Generation
         ↓
    Claim Support Verification
         ↓
Answer + Sources / Fallback Excerpt
```

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- 16GB RAM minimum (32GB recommended)
- 100GB free disk space
- CPU-optimized (no GPU required!)

### Quick Setup

```bash
# Clone the repository
cd trace-tutor

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download local model (choose one)
# Option 1: Ollama (recommended)
ollama pull phi3:3b-mini-instruct

# Option 2: Alternative models
# ollama pull llama3.2:3b-instruct
# ollama pull qwen2.5:3b-instruct

# Setup NCERT corpus
python scripts/download_ncert.py --grades 3 4 5 --subject evs

# Build initial evidence bundles
python scripts/build_corpus.py --config configs/study.yaml

# Build retrieval indices
python scripts/build_index.py

# Run baseline experiment
python scripts/run_experiment.py --config configs/baseline.yaml
```

## 🎮 Usage

### Interactive Demo

```python
from trace_tutor import TRACETutor

# Initialize system
tutor = TRACETutor(config_path="configs/study.yaml")

# Ask a question
response = tutor.ask(
    question="Why do plants need sunlight?",
    grade=4,
    subject="EVS"
)

print(f"Answer: {response.answer}")
print(f"Sources: {response.source_ids}")
print(f"Evidence used: {len(response.evidence_bundles)}")
```

### API Mode (Optional)

```bash
# Start FastAPI server
python -m trace_tutor.api --port 8000

# Query endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Why do plants need sunlight?", "grade": 4}'
```

### With API Key Support

```python
# Use OpenAI/Anthropic instead of local model
tutor = TRACETutor(
    config_path="configs/study.yaml",
    api_provider="openai",
    api_key="your-key-here"
)
```

## 🧪 Experiments & Evaluation

### Run Ablations

```bash
# Full ablation study (B0, B1, A1-A4, Full)
python scripts/run_ablations.py --config configs/experiment_trace.yaml

# Results saved to: runs/ablation_YYYYMMDD_HHMMSS/
```

### Evaluate Specific Configuration

```bash
python scripts/evaluate.py \
  --config configs/baseline.yaml \
  --gold-data data/gold/questions_pilot.jsonl \
  --output runs/eval_baseline/
```

### Human Evaluation

```bash
# Generate comparison pairs for annotation
python scripts/prepare_human_eval.py \
  --run1 runs/baseline/ \
  --run2 runs/full_trace/ \
  --output data/human_eval/pairs.jsonl

# Run annotation interface
streamlit run scripts/human_eval_ui.py
```

## 📊 Results & Reproducibility

All experiments are fully reproducible with:
- Fixed random seeds
- Locked model versions
- Frozen corpus snapshots
- Complete run manifests

Example results structure:
```
runs/
├── baseline_20260105_143022/
│   ├── config.yaml
│   ├── ranks.jsonl
│   ├── answers.jsonl
│   ├── metrics.json
│   └── manifest.json
```

## 📚 Dataset

### NCERT Corpus (Grades 3-5, EVS)
- **Scope**: Environmental Studies, Grades 3-5
- **Chapters**: 20-30 selected chapters
- **Evidence Bundles**: 1,000-3,000 validated items
- **Learning Outcomes**: 80-150 mapped
- **Prerequisite Edges**: 150-250 expert-reviewed

### Gold Questions
- **Pilot**: 300 questions
- **Final**: 600-800 questions
- **Types**: Direct fact, conceptual, diagram-dependent, prerequisite gap
- **Annotation**: Expert-reviewed with role labels

### Data Rights
All NCERT materials used under authorized educational use. See `docs/data_rights.md` for complete licensing information.

## 🔬 Research Contributions

1. **Curriculum Evidence Bundles**: Source-preserving instructional units
2. **Prerequisite Licensing**: Verified ancestor-based lower-grade admission
3. **PS-Evidence Selector**: Role-constrained minimal evidence optimization
4. **TRACE Verification**: Independent claim support checking
5. **No-Expert Methodology**: Conservative approach for solo researchers

## 📖 Citation

```bibtex
@inproceedings{trace-tutor-2026,
  title={TRACE-Tutor: Role-Constrained Evidence Selection for Curriculum-Grounded Tutoring},
  author={[Your Name]},
  booktitle={IEEE Transactions on Learning Technologies / AIED 2027},
  year={2026}
}
```

## 🗺️ Roadmap

### ✅ Core (Tier 1 - Current)
- [x] Evidence bundle schema
- [x] Hybrid retrieval (BM25 + Dense + RRF)
- [x] PS-Evidence selector
- [x] Local LLM generation
- [x] Verification module
- [x] Ablation framework
- [x] Basic accessibility

### 🔄 Extended (Tier 2 - Future)
- [ ] Expert prerequisite graph validation
- [ ] Hindi/Marathi multilingual support
- [ ] Visual retrieval with CLIP
- [ ] Teacher usability study
- [ ] Fine-tuned visual models
- [ ] Student pilot study

### 🌟 Full Platform (Tier 3 - Research Program)
- [ ] Formal differential privacy
- [ ] Federated learning
- [ ] Mobile application
- [ ] Classroom deployment
- [ ] Longitudinal outcomes

## 📝 Paper Status

**Target Venue**: IEEE Transactions on Learning Technologies (TLT) or AIED 2027

**Sections**:
1. ✅ Introduction & Problem
2. ✅ Related Work
3. ✅ Method Formulation
4. 🔄 Experiments (in progress)
5. ⏳ Results (after ablations)
6. ⏳ Error Analysis
7. ⏳ Limitations & Discussion

## 🤝 Contributing

This is a research prototype. For improvements:
1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/`
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- NCERT for educational materials
- HuggingFace for model infrastructure
- Research papers: KnowSTU, SAMAC-R3-MED, KICNLE, Adenuga (2026)

## 📧 Contact

For research collaboration or questions: [your-email@example.com]

---

**Status**: Research prototype - active development  
**Version**: 1.0.0  
**Last Updated**: January 2026  
**Research Phase**: Tier 1 (Core System)
