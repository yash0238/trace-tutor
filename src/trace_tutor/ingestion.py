"""
PDF ingestion and evidence bundle construction.

Extracts text, figures, and structure from NCERT textbooks while preserving
complete provenance metadata.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import pymupdf  # PyMuPDF
from PIL import Image
import hashlib

from trace_tutor.schema import Evidence, Role


logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Extracted content from a single page."""
    page_num: int
    text: str
    blocks: List[Dict]
    images: List[Dict]
    tables: List[Dict]


class NCERTExtractor:
    """
    Extract structured content from NCERT PDF textbooks.
    
    Preserves page structure, figures with captions, and exercises.
    """
    
    def __init__(self, output_dir: str = "data/interim"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
    
    def extract_pdf(self, pdf_path: str, source_id: str) -> List[PageContent]:
        """
        Extract all content from PDF with structure preservation.
        
        Args:
            pdf_path: Path to NCERT PDF file
            source_id: Unique source identifier (e.g., 'ncert_evs_grade4')
        
        Returns:
            List of PageContent objects
        """
        logger.info(f"Extracting PDF: {pdf_path}")
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        pages = []
        
        with pymupdf.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks with position
                blocks = page.get_text("dict")["blocks"]
                
                # Extract text (try layout-preserving first, fallback to OCR)
                text = page.get_text("text", sort=True)
                if not text.strip() and page.get_images():
                    # Scanned page - use OCR
                    logger.debug(f"Using OCR for page {page_num + 1}")
                    textpage = page.get_textpage_ocr()
                    text = page.get_text(textpage=textpage)
                
                # Extract images
                images = self._extract_images(page, page_num, source_id)
                
                # Detect tables (basic heuristic)
                tables = self._detect_tables(blocks)
                
                pages.append(PageContent(
                    page_num=page_num + 1,
                    text=text,
                    blocks=blocks,
                    images=images,
                    tables=tables
                ))
        
        logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
        return pages
    
    def _extract_images(
        self,
        page: pymupdf.Page,
        page_num: int,
        source_id: str
    ) -> List[Dict]:
        """Extract and save images from page."""
        images = []
        
        for img_idx, img_info in enumerate(page.get_images()):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Generate unique filename
                    img_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                    filename = f"{source_id}_p{page_num + 1:03d}_img{img_idx:02d}_{img_hash}.{image_ext}"
                    filepath = self.figures_dir / filename
                    
                    # Save image
                    filepath.write_bytes(image_bytes)
                    
                    # Get image dimensions
                    with Image.open(filepath) as im:
                        width, height = im.size
                    
                    images.append({
                        "figure_id": f"{source_id}_p{page_num + 1:03d}_fig{img_idx:02d}",
                        "path": str(filepath),
                        "bbox": img_info[1:5] if len(img_info) > 1 else None,
                        "width": width,
                        "height": height,
                        "format": image_ext
                    })
            
            except Exception as e:
                logger.warning(f"Failed to extract image {img_idx} from page {page_num + 1}: {e}")
        
        return images
    
    def _detect_tables(self, blocks: List[Dict]) -> List[Dict]:
        """Detect potential tables based on block layout."""
        # Simplified table detection
        # TODO: Implement more sophisticated table detection
        tables = []
        return tables
    
    def save_pages(self, pages: List[PageContent], source_id: str):
        """Save extracted pages to JSONL."""
        output_file = self.output_dir / f"{source_id}_pages.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for page in pages:
                record = {
                    "source_id": source_id,
                    "page": page.page_num,
                    "text": page.text,
                    "num_blocks": len(page.blocks),
                    "num_images": len(page.images),
                    "images": page.images
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved pages to {output_file}")
        return output_file


class BundleBuilder:
    """
    Construct Evidence bundles from extracted pages.
    
    Creates pedagogically meaningful units with role labels and provenance.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_length = config.get("min_bundle_length", 32)
        self.max_length = config.get("max_bundle_length", 512)
    
    def build_from_pages(
        self,
        pages: List[PageContent],
        source_id: str,
        grade: int,
        subject: str,
        chapter_mapping: Optional[Dict] = None
    ) -> List[Evidence]:
        """
        Build evidence bundles from extracted pages.
        
        Args:
            pages: Extracted page content
            source_id: Source document ID
            grade: Target grade level
            subject: Subject area
            chapter_mapping: Optional mapping of page ranges to chapters
        
        Returns:
            List of Evidence bundles
        """
        bundles = []
        
        for page_content in pages:
            # Split page into paragraphs
            paragraphs = self._split_paragraphs(page_content.text)
            
            # Determine chapter ID
            chapter_id = self._get_chapter_id(
                page_content.page_num,
                chapter_mapping
            )
            
            # Create bundles from paragraphs
            for para_idx, para_text in enumerate(paragraphs):
                if len(para_text.split()) < 10:  # Skip very short paragraphs
                    continue
                
                # Infer basic roles (requires manual review for accuracy)
                roles = self._infer_roles(para_text)
                
                # Check if paragraph has associated figure
                figure_info = self._find_nearby_figure(
                    page_content,
                    para_idx,
                    len(paragraphs)
                )
                
                bundle_id = f"{source_id}_p{page_content.page_num:03d}_a{para_idx:02d}"
                atom_id = f"{source_id}:p{page_content.page_num}:span{para_idx}"
                
                bundle = Evidence(
                    id=bundle_id,
                    source_id=source_id,
                    page=page_content.page_num,
                    chapter_id=chapter_id or "unknown",
                    grade=grade,
                    subject=subject,
                    roles=roles,
                    text=para_text,
                    source_atom_ids=[atom_id],
                    figure_id=figure_info["figure_id"] if figure_info else None,
                    figure_caption=figure_info.get("caption") if figure_info else None,
                    figure_path=figure_info.get("path") if figure_info else None,
                    language="en",
                    rights_status="authorized_local_only",
                    quality_checked=False
                )
                
                bundles.append(bundle)
        
        logger.info(f"Built {len(bundles)} evidence bundles from {len(pages)} pages")
        return bundles
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Simple splitting by double newlines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Further split by single newlines if paragraphs are too long
        result = []
        for para in paragraphs:
            if len(para.split()) > self.max_length:
                # Split on periods or newlines
                sentences = para.split('\n')
                result.extend([s.strip() for s in sentences if s.strip()])
            else:
                result.append(para)
        
        return result
    
    def _infer_roles(self, text: str) -> set[Role]:
        """
        Infer instructional roles from text (heuristic-based).
        
        NOTE: This is a starting point. Manual review and correction
        are essential for research quality.
        """
        roles: set[Role] = set()
        
        text_lower = text.lower()
        
        # Concept indicators
        if any(word in text_lower for word in [
            "is called", "means", "definition", "what is", "refers to", "known as"
        ]):
            roles.add("concept")
        
        # Example indicators
        if any(word in text_lower for word in [
            "for example", "for instance", "such as", "like", "e.g."
        ]):
            roles.add("example")
        
        # Exercise indicators
        if any(word in text_lower for word in [
            "question", "exercise", "answer", "solve", "find", "calculate"
        ]) or text.strip().endswith("?"):
            roles.add("exercise")
        
        # Default to concept if no specific role detected
        if not roles:
            roles.add("concept")
        
        return roles
    
    def _get_chapter_id(
        self,
        page_num: int,
        chapter_mapping: Optional[Dict]
    ) -> Optional[str]:
        """Get chapter ID for a page number."""
        if not chapter_mapping:
            return None
        
        # Handle two formats:
        # 1. {page_num: chapter_id} - direct mapping
        # 2. {chapter_id: (start, end)} - range mapping
        if page_num in chapter_mapping:
            return chapter_mapping[page_num]
        
        for chapter_id, page_range in chapter_mapping.items():
            if isinstance(page_range, (list, tuple)) and len(page_range) == 2:
                start, end = page_range
                if start <= page_num <= end:
                    return chapter_id
        
        return None
    
    def _find_nearby_figure(
        self,
        page_content: PageContent,
        para_idx: int,
        total_paragraphs: int
    ) -> Optional[Dict]:
        """Find figure associated with paragraph."""
        # Simple heuristic: if page has images, associate with middle paragraphs
        if not page_content.images:
            return None
        
        # Associate first image with first half of page content
        if para_idx < total_paragraphs // 2 and page_content.images:
            return page_content.images[0]
        
        return None
    
    def save_bundles(self, bundles: List[Evidence], output_file: str):
        """Save evidence bundles to JSONL."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for bundle in bundles:
                f.write(bundle.model_dump_json() + '\n')
        
        logger.info(f"Saved {len(bundles)} bundles to {output_file}")
    
    @staticmethod
    def load_bundles(input_file: str) -> List[Evidence]:
        """Load evidence bundles from JSONL."""
        bundles = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    bundle = Evidence.model_validate_json(line)
                    bundles.append(bundle)
        
        logger.info(f"Loaded {len(bundles)} bundles from {input_file}")
        return bundles


def validate_bundles(bundles: List[Evidence]) -> Tuple[List[Evidence], List[str]]:
    """
    Validate evidence bundles and return valid ones + error messages.
    
    Returns:
        (valid_bundles, error_messages)
    """
    valid = []
    errors = []
    
    seen_ids = set()
    
    for bundle in bundles:
        # Check for duplicate IDs
        if bundle.id in seen_ids:
            errors.append(f"Duplicate bundle ID: {bundle.id}")
            continue
        seen_ids.add(bundle.id)
        
        # Check required fields
        if not bundle.roles:
            errors.append(f"Bundle {bundle.id} has no roles")
            continue
        
        if not bundle.source_atom_ids:
            errors.append(f"Bundle {bundle.id} has no source atoms")
            continue
        
        if len(bundle.text.split()) < 10:
            errors.append(f"Bundle {bundle.id} text too short")
            continue
        
        # Check figure metadata consistency
        if bundle.figure_id and not bundle.figure_caption:
            errors.append(f"Bundle {bundle.id} has figure but no caption")
            continue
        
        valid.append(bundle)
    
    logger.info(f"Validated {len(valid)}/{len(bundles)} bundles, {len(errors)} errors")
    
    return valid, errors


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Extract PDF
    extractor = NCERTExtractor()
    # pages = extractor.extract_pdf("data/raw/ncert_evs_grade4.pdf", "ncert_evs_g4")
    # extractor.save_pages(pages, "ncert_evs_g4")
    
    # Build bundles
    # builder = BundleBuilder({"min_bundle_length": 32, "max_bundle_length": 512})
    # bundles = builder.build_from_pages(pages, "ncert_evs_g4", grade=4, subject="EVS")
    # builder.save_bundles(bundles, "data/bundles/ncert_evs_g4_bundles.jsonl")
    
    print("Ingestion module ready. Use scripts/build_corpus.py for batch processing.")
