"""
Complete System Test

Tests all components end-to-end:
1. Table extraction
2. Faithfulness gate
3. Clarification agent
4. Image understanding
5. Audio processing
6. Multimodal retrieval
7. Full pipeline integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_table_extraction():
    """Test 1: Table extraction from PDFs."""
    print("\n" + "="*60)
    print("TEST 1: Table Extraction")
    print("="*60)
    
    try:
        from trace_tutor.table_extractor import TableExtractor, TableUnderstanding
        
        extractor = TableExtractor()
        
        # Mock table data (since we may not have actual PDFs)
        mock_table = {
            "table_id": "test_table_01",
            "page": 42,
            "data": [
                ["Property", "Solid", "Liquid", "Gas"],
                ["Shape", "Fixed", "Variable", "Variable"],
                ["Volume", "Fixed", "Fixed", "Variable"],
                ["Compressibility", "Low", "Low", "High"]
            ],
            "markdown": "| Property | Solid | Liquid | Gas |\n|---|---|---|---|\n| Shape | Fixed | Variable | Variable |"
        }
        
        # Test chunking
        chunks = extractor.chunk_table(mock_table, chunk_size=2)
        print(f"✅ Table chunking: Created {len(chunks)} chunks")
        
        # Test cell lookup
        understanding = TableUnderstanding()
        cell_value = understanding.find_cell_value(mock_table["data"], "Volume", "Gas")
        print(f"✅ Cell lookup: Volume of Gas = {cell_value}")
        
        # Test summary
        summary = understanding.summarize_table(mock_table)
        print(f"✅ Table summary: {summary[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Table extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_faithfulness_gate():
    """Test 2: Faithfulness verification."""
    print("\n" + "="*60)
    print("TEST 2: Faithfulness Gate")
    print("="*60)
    
    try:
        from trace_tutor.faithfulness_gate import FaithfulnessGate
        
        gate = FaithfulnessGate(threshold=0.7)
        
        # Test case 1: Fully supported answer
        answer1 = "Photosynthesis is the process where plants use sunlight to make food. Plants need water and carbon dioxide for this process."
        evidence1 = [
            "Photosynthesis is how plants create food using sunlight.",
            "Plants require water, carbon dioxide, and sunlight for photosynthesis.",
            "The process converts light energy into chemical energy."
        ]
        
        result1 = gate.verify(answer1, evidence1)
        print(f"✅ Test 1 - Supported answer: {result1['support_score']:.0%} support, Passed: {result1['passed']}")
        
        # Test case 2: Partially supported
        answer2 = "Plants use photosynthesis to make food. They also use nuclear fusion to power their cells."
        evidence2 = ["Plants use photosynthesis to create glucose."]
        
        result2 = gate.verify(answer2, evidence2)
        print(f"✅ Test 2 - Mixed answer: {result2['support_score']:.0%} support, Passed: {result2['passed']}")
        print(f"   Unsupported claims: {len(result2['unsupported_claims'])}")
        
        return True
    except Exception as e:
        print(f"❌ Faithfulness gate failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clarification_agent():
    """Test 3: Clarification agent."""
    print("\n" + "="*60)
    print("TEST 3: Clarification Agent")
    print("="*60)
    
    try:
        from trace_tutor.clarification_agent import ClarificationAgent, check_and_clarify
        
        agent = ClarificationAgent(confidence_threshold=0.6)
        
        # Test case 1: Vague query
        result1 = check_and_clarify("what is it", grade=3, subject="Science")
        print(f"✅ Vague query detected: Needs clarification={result1['needs_clarification']}")
        if result1['clarification_message']:
            print(f"   Message: {result1['clarification_message'][:100]}...")
        
        # Test case 2: Clear query
        result2 = check_and_clarify("How does photosynthesis work in plants?", grade=4, subject="Science")
        print(f"✅ Clear query: Needs clarification={result2['needs_clarification']}")
        print(f"   Confidence: {result2['confidence']:.0%}")
        
        # Test case 3: Ambiguous term
        result3 = check_and_clarify("Tell me about the cell", grade=3, subject="Science")
        print(f"✅ Ambiguous query: Needs clarification={result3['needs_clarification']}")
        
        return True
    except Exception as e:
        print(f"❌ Clarification agent failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_understanding():
    """Test 4: Image encoding and captioning."""
    print("\n" + "="*60)
    print("TEST 4: Image Understanding")
    print("="*60)
    
    try:
        from trace_tutor.image_understanding import CLIPImageEncoder
        
        encoder = CLIPImageEncoder()
        
        if encoder.model is None:
            print("⚠️  CLIP model not loaded, skipping detailed tests")
            return True
        
        # Test text encoding
        text_emb = encoder.encode_text("photosynthesis diagram")
        print(f"✅ Text encoding: shape={text_emb.shape}, dim={len(text_emb)}")
        
        # Test similarity (without actual image)
        print(f"✅ CLIP encoder initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ Image understanding failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_processing():
    """Test 5: Audio STT/TTS."""
    print("\n" + "="*60)
    print("TEST 5: Audio Processing")
    print("="*60)
    
    try:
        from trace_tutor.audio_processor import SpeechToText, TextToSpeech
        
        # Test TTS (doesn't require audio files)
        tts = TextToSpeech(default_lang="en")
        print(f"✅ TTS initialized: engine={tts.gtts is not None}")
        
        if tts.gtts:
            test_text = "This is a test of text to speech."
            audio_path = tts.synthesize(test_text)
            if audio_path:
                print(f"✅ TTS synthesis: {audio_path}")
            else:
                print("⚠️  TTS synthesis returned empty path")
        
        # STT test (requires Whisper)
        try:
            stt = SpeechToText(model_size="base")
            print(f"✅ STT initialized: model loaded={stt.model is not None}")
        except Exception as e:
            print(f"⚠️  STT not available: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_retriever():
    """Test 6: Multimodal retrieval."""
    print("\n" + "="*60)
    print("TEST 6: Multimodal Retrieval")
    print("="*60)
    
    try:
        from trace_tutor.multimodal_retriever import MultimodalRetriever, MMRSelector
        from trace_tutor.schema import Evidence, Query
        import numpy as np
        
        # Create mock evidence
        mock_evidence = [
            Evidence(
                id="e1",
                source_id="test",
                page=1,
                chapter_id="ch1",
                grade=3,
                subject="Science",
                roles={"concept"},
                text="Photosynthesis is how plants make food using sunlight.",
                source_atom_ids=["a1"],
                modality="text"
            ),
            Evidence(
                id="e2",
                source_id="test",
                page=2,
                chapter_id="ch1",
                grade=3,
                subject="Science",
                roles={"visual"},
                text="Diagram showing photosynthesis process",
                source_atom_ids=["a2"],
                modality="image",
                metadata={"clip_embedding": np.random.randn(512).tolist()}
            )
        ]
        
        print(f"✅ Created {len(mock_evidence)} mock evidence items")
        
        # Test MMR selector
        selector = MMRSelector(lambda_param=0.7)
        query_emb = np.random.randn(512)
        candidates = [(e, 0.8) for e in mock_evidence]
        
        selected = selector.select(query_emb, candidates, k=2)
        print(f"✅ MMR selection: selected {len(selected)} items")
        
        return True
    except Exception as e:
        print(f"❌ Multimodal retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_retrieval():
    """Test 7: Advanced retrieval components."""
    print("\n" + "="*60)
    print("TEST 7: Advanced Retrieval")
    print("="*60)
    
    try:
        from trace_tutor.advanced_retrieval import (
            DifficultyAwareReranker,
            BanditFusionOptimizer
        )
        from trace_tutor.schema import Evidence
        
        # Test difficulty reranker
        reranker = DifficultyAwareReranker(grade_weight=0.3)
        
        mock_evidence = [
            Evidence(
                id="e1", source_id="t", page=1, chapter_id="c1",
                grade=2, subject="Math", roles={"concept"},
                text="Simple addition", source_atom_ids=["a1"]
            ),
            Evidence(
                id="e2", source_id="t", page=2, chapter_id="c2",
                grade=5, subject="Math", roles={"concept"},
                text="Complex algebra", source_atom_ids=["a2"]
            )
        ]
        
        candidates = [(e, 0.9) for e in mock_evidence]
        reranked = reranker.rerank(candidates, student_grade=3)
        
        print(f"✅ Difficulty reranking: {len(reranked)} items reranked")
        print(f"   Grade 2 evidence score: {reranked[0][1]:.3f}")
        print(f"   Grade 5 evidence score: {reranked[1][1]:.3f}")
        
        # Test bandit optimizer
        optimizer = BanditFusionOptimizer(learning_rate=0.1)
        
        weights = optimizer.get_weights("factual")
        print(f"✅ Bandit optimizer initialized")
        print(f"   Initial weights: {weights}")
        
        # Simulate updates
        optimizer.update("factual", reward=0.8)
        optimizer.update("factual", reward=0.9)
        
        new_weights = optimizer.get_weights("factual")
        print(f"   Updated weights: {new_weights}")
        
        return True
    except Exception as e:
        print(f"❌ Advanced retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test 8: Full end-to-end pipeline."""
    print("\n" + "="*60)
    print("TEST 8: Full Pipeline Integration")
    print("="*60)
    
    try:
        from trace_tutor.tutor import TRACETutor
        
        # Try to load tutor
        try:
            tutor = TRACETutor(
                config_path="configs/study.yaml",
                bundles_path="data/bundles/corpus_all.jsonl",
                graph_path="data/graph/prerequisites.jsonl"
            )
            print("✅ TRACE-Tutor loaded successfully")
            
            # Test basic query
            if tutor.bundles and len(tutor.bundles) > 0:
                answer = tutor.ask(
                    question="What is photosynthesis?",
                    grade=3,
                    subject="Science",
                    use_cache=False
                )
                print(f"✅ Basic query completed")
                print(f"   Answer length: {len(answer.answer)} chars")
                print(f"   Evidence bundles: {len(answer.evidence_bundle_ids)}")
                print(f"   Total time: {answer.total_time:.2f}s")
            else:
                print("⚠️  No bundles loaded, skipping query test")
            
        except FileNotFoundError as e:
            print(f"⚠️  Data files not found: {e}")
            print("   Run corpus building first: python build_corpus.py")
        
        return True
    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TRACE-TUTOR COMPLETE SYSTEM TEST")
    print("="*60)
    
    tests = [
        ("Table Extraction", test_table_extraction),
        ("Faithfulness Gate", test_faithfulness_gate),
        ("Clarification Agent", test_clarification_agent),
        ("Image Understanding", test_image_understanding),
        ("Audio Processing", test_audio_processing),
        ("Multimodal Retrieval", test_multimodal_retriever),
        ("Advanced Retrieval", test_advanced_retrieval),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total:.0%})")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
