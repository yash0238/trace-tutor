# Data Rights and Licensing

**Document Version**: 1.0  
**Last Updated**: January 5, 2026  
**Status**: Research Use - Local Only

## Overview

This document tracks the copyright, licensing, and usage permissions for all data sources used in the TRACE-Tutor research project.

---

## NCERT Textbooks

### Source Information
- **Provider**: National Council of Educational Research and Training (NCERT), India
- **URL**: https://ncert.nic.in/textbook.php
- **Content Type**: Educational textbooks (PDF)
- **Subjects Used**: Environmental Studies (EVS)
- **Grades Used**: 3, 4, 5
- **Access Date**: January 2026

### License Terms
NCERT materials are provided for educational purposes under the following terms:

1. **Permitted Use**:
   - Educational and research use
   - Non-commercial academic purposes
   - Citation with proper attribution

2. **Restrictions**:
   - Commercial use prohibited without permission
   - Adaptation/translation requires authorization
   - Redistribution subject to terms
   - No warranty implied

3. **Reference**:
   - NCERT e-content license: https://ncert.nic.in/
   - Terms of use: https://ncert.nic.in/terms-conditions.php

### Our Usage
- **Purpose**: Research on AI tutoring systems
- **Scope**: Local processing, evidence extraction, retrieval indexing
- **Storage**: Private research environment, not publicly redistributed
- **Attribution**: Full credit given in all publications

### Public Release Strategy
For reproducible research release, we will provide:
- ✅ Evidence bundle metadata (IDs, roles, LOs, page numbers)
- ✅ Gold question labels and annotations
- ✅ Prerequisite graph structure
- ✅ All source code and evaluation scripts
- ✅ Reconstruction instructions
- ❌ Raw PDF files (users must download from NCERT)
- ❌ Extracted text spans exceeding fair use
- ❌ Copyrighted figures without permission

---

## NCERT Learning Outcomes

### Source
- **Document**: Learning Outcomes at Elementary Stage
- **URL**: https://ncert.nic.in/learning-outcome.php
- **License**: Official NCERT publication for educational use

### Our Usage
- Mapping curriculum to evidence bundles
- Prerequisite graph construction
- Evaluation framework alignment

---

## Third-Party Models and Tools

### Sentence Transformers
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **License**: Apache 2.0
- **Source**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Usage**: Dense text embedding for retrieval

### Cross-Encoder Reranker
- **Model**: `cross-encoder/ms-marco-MiniLM-L6-v2`
- **License**: Apache 2.0
- **Source**: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
- **Usage**: Candidate reranking

### Local LLM (Primary)
- **Model**: Phi-3 Mini 3B Instruct
- **License**: MIT License
- **Source**: https://huggingface.co/microsoft/phi-3-mini-3b-instruct
- **Usage**: Grounded answer generation

### Alternative Models (Optional)
- **Llama 3.2 3B**: Meta Community License
- **Qwen 2.5 3B**: Apache 2.0
- **Gemma 2B**: Gemma Terms of Use

### NLI Verifier
- **Model**: `microsoft/deberta-v3-base-mnli`
- **License**: MIT License
- **Source**: https://huggingface.co/microsoft/deberta-v3-base
- **Usage**: Claim support verification

---

## Research Annotations

### Gold Questions
- **Creator**: Research team
- **Status**: Original work
- **License**: MIT (to be released with paper)
- **Content**: Question text, target grades, gold evidence IDs, role labels

### Prerequisite Graph
- **Creator**: Research team (based on NCERT curriculum)
- **Status**: Derived work, expert-reviewable structure
- **License**: MIT (structure only; nodes reference NCERT LOs)
- **Content**: LO IDs, prerequisite edges, review status

### Evidence Bundle Annotations
- **Creator**: Research team
- **Status**: Metadata layer over NCERT content
- **License**: Metadata (roles, IDs, provenance) - MIT
- **Content**: Bundle IDs, role labels, page mappings, LO links

---

## Publication and Sharing

### What We Will Share
1. **Code** (MIT License):
   - All Python modules
   - Preprocessing scripts
   - Evaluation harness
   - Experiment configs

2. **Metadata and Labels** (MIT License):
   - Evidence bundle schema and IDs
   - Gold question annotations
   - Prerequisite graph structure
   - Evaluation results

3. **Documentation**:
   - Complete setup guide
   - Data reconstruction instructions
   - Experiment protocols
   - Paper and supplementary materials

### What Users Must Obtain
1. **NCERT Textbooks**:
   - Download from official NCERT website
   - Comply with NCERT terms of use
   - Use for educational/research purposes

2. **Pre-trained Models**:
   - Download from HuggingFace Hub
   - Comply with respective model licenses
   - Accept model use terms

---

## Compliance Checklist

- [x] Source inventory complete
- [x] License terms reviewed
- [x] Attribution plan documented
- [x] Public release strategy defined
- [x] Fair use boundaries established
- [ ] Legal counsel review (if institutional release)
- [ ] Written permission obtained (if needed for figures)

---

## Contact and Permissions

For questions about data usage or permissions:
1. **NCERT Materials**: Contact NCERT directly
2. **Research Annotations**: Contact research team
3. **Model Licenses**: See respective HuggingFace pages

---

## Updates and Changes

### Version History
- **v1.0 (2026-01-05)**: Initial data rights documentation

### Change Log
- Added NCERT terms clarification
- Documented public release strategy
- Listed all third-party model licenses

---

## Declaration

**Research Use Statement**: All materials in this research project are used for non-commercial academic research purposes only. We do not claim ownership of any NCERT content. All textbook materials remain the property of NCERT and the Government of India.

**Attribution Commitment**: All publications will properly cite NCERT materials and acknowledge all open-source tools and models used.

**Compliance Commitment**: We will respect all license terms and will not publicly redistribute materials beyond fair use and permitted sharing.

---

**Last Review**: January 5, 2026  
**Next Review**: Before public release or paper submission  
**Maintained by**: TRACE-Tutor Research Team
