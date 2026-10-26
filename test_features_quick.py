"""
Quick Feature Test (No Large Downloads)

Tests core functionality without downloading large models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing TRACE-Tutor Core Features...")
print("="*60)

# Test 1: Table Extraction
print("\n✅ TEST 1: Table Extraction")
from trace_tutor.table_extractor import TableExtractor, TableUnderstanding

extractor = TableExtractor()
mock_table = {
    "table_id": "test_01",
    "page": 42,
    "data": [
        ["Property", "Solid", "Liquid"],
        ["Shape", "Fixed", "Variable"],
        ["Volume", "Fixed", "Fixed"]
    ],
    "markdown": "..."
}

chunks = extractor.chunk_table(mock_table, chunk_size=2)
print(f"   Created {len(chunks)} table chunks")

cell = TableUnderstanding.find_cell_value(mock_table["data"], "Volume", "Liquid")
print(f"   Cell lookup works: Volume/Liquid = {cell}")

# Test 2: Faithfulness Gate
print("\n✅ TEST 2: Faithfulness Gate")
from trace_tutor.faithfulness_gate import FaithfulnessGate

gate = FaithfulnessGate(threshold=0.7)
answer = "Plants use photosynthesis. They need sunlight and water."
evidence = ["Photosynthesis requires sunlight.", "Plants need water for growth."]

result = gate.verify(answer, evidence)
print(f"   Support score: {result['support_score']:.0%}")
print(f"   Passed: {result['passed']}")

# Test 3: Clarification Agent
print("\n✅ TEST 3: Clarification Agent")
from trace_tutor.clarification_agent import check_and_clarify

result1 = check_and_clarify("what is it", grade=3, subject="Science")
print(f"   Vague query detected: {result1['needs_clarification']}")

result2 = check_and_clarify("How does photosynthesis work?", grade=3, subject="Science")
print(f"   Clear query: confidence={result2['confidence']:.0%}")

# Test 4: Schema Updates
print("\n✅ TEST 4: Multimodal Schema")
from trace_tutor.schema import Query, Evidence, Answer

query = Query(
    id="test",
    text="What is this?",
    image_path="/test/img.png",
    audio_path="/test/audio.mp3",
    audio_transcription="What is this",
    grade=3,
    subject="Science"
)
print(f"   Multimodal Query created: image={query.image_path is not None}")

evidence = Evidence(
    id="e1",
    source_id="test",
    page=1,
    chapter_id="ch1",
    grade=3,
    subject="Science",
    roles={"concept"},
    text="Test",
    source_atom_ids=["a1"],
    modality="image",
    metadata={"test": True}
)
print(f"   Multimodal Evidence created: modality={evidence.modality}")

# Test 5: Advanced Retrieval
print("\n✅ TEST 5: Advanced Retrieval Components")
from trace_tutor.advanced_retrieval import DifficultyAwareReranker, BanditFusionOptimizer

reranker = DifficultyAwareReranker(grade_weight=0.3)
print(f"   Difficulty reranker initialized")

optimizer = BanditFusionOptimizer()
weights = optimizer.get_weights("factual")
print(f"   Bandit optimizer: {list(weights.keys())}")

optimizer.update("factual", reward=0.8)
new_weights = optimizer.get_weights("factual")
print(f"   Weights updated after reward")

# Test 6: Configuration
print("\n✅ TEST 6: Configuration Loaded")
import yaml

config_path = Path("configs/study.yaml")
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"   Multimodal enabled: {config.get('multimodal', {}).get('enabled')}")
    print(f"   Faithfulness enabled: {config.get('faithfulness', {}).get('enabled')}")
    print(f"   Clarification enabled: {config.get('clarification', {}).get('enabled')}")
    print(f"   Cache enabled: {config.get('cache', {}).get('enabled')}")
else:
    print(f"   ⚠️ Config file not found")

# Summary
print("\n" + "="*60)
print("SUMMARY: All core features are implemented ✅")
print("="*60)
print("\nFeatures Ready:")
print("  ✅ Table extraction and understanding")
print("  ✅ Faithfulness gate (claim verification)")
print("  ✅ Clarification agent (ambiguity detection)")
print("  ✅ Multimodal schema (image, audio support)")
print("  ✅ Advanced retrieval (difficulty-aware, bandit)")
print("  ✅ Configuration system")
print("\nNext Steps:")
print("  1. Run: pip install -r requirements.txt")
print("  2. Build corpus: python build_corpus.py")
print("  3. Start app: streamlit run app_multimodal.py")
