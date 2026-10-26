"""
Image Understanding Module

Implements cross-modal retrieval using CLIP and image captioning for TRACE-Tutor.
Provides image encoding, caption generation, and visual evidence extraction.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
from PIL import Image
import torch

from trace_tutor.schema import Evidence

logger = logging.getLogger(__name__)


class CLIPImageEncoder:
    """
    CLIP-based image-text encoder for cross-modal retrieval.
    
    Uses 'clip-ViT-B-32-multilingual-v1' for Hindi/Urdu/English support.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1", device: str = "cpu"):
        """
        Initialize CLIP encoder.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model_name = model_name
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading CLIP model: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = 512  # CLIP ViT-B/32 outputs 512-dim
            logger.info(f"CLIP encoder ready (dim={self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self.model = None
            self.embedding_dim = 512
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode image to CLIP embedding.
        
        Args:
            image_path: Path to image file
            
        Returns:
            512-dimensional embedding vector
        """
        if self.model is None:
            logger.warning("CLIP model not loaded, returning zero vector")
            return np.zeros(self.embedding_dim)
        
        try:
            img = Image.open(image_path).convert('RGB')
            embedding = self.model.encode(img, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return np.zeros(self.embedding_dim)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to CLIP embedding (shared space with images).
        
        Args:
            text: Text query
            
        Returns:
            512-dimensional embedding vector
        """
        if self.model is None:
            logger.warning("CLIP model not loaded, returning zero vector")
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return np.zeros(self.embedding_dim)
    
    def compute_similarity(self, text: str, image_path: str) -> float:
        """
        Compute cosine similarity between text and image.
        
        Args:
            text: Text query
            image_path: Path to image
            
        Returns:
            Similarity score [0, 1]
        """
        text_emb = self.encode_text(text)
        image_emb = self.encode_image(image_path)
        
        # Cosine similarity
        similarity = np.dot(text_emb, image_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(image_emb) + 1e-8)
        return float(similarity)


class ImageCaptionGenerator:
    """
    Generate captions for images using Gemini Vision API.
    
    Provides caption-mediated retrieval as fallback when CLIP is unavailable.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        """
        Initialize caption generator.
        
        Args:
            api_key: Gemini API key (reads from env if None)
            model: Gemini model identifier
        """
        self.model_name = model
        self.api_key = api_key
        self.client = None
        
        if api_key or self._load_api_key():
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key or self._load_api_key())
                self.client = genai.GenerativeModel(model)
                logger.info(f"Gemini Vision API ready ({model})")
            except Exception as e:
                logger.warning(f"Gemini Vision API unavailable: {e}")
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment."""
        import os
        return os.getenv("GEMINI_API_KEY")
    
    def generate_caption(self, image_path: str, detail_level: str = "concise") -> str:
        """
        Generate caption for image.
        
        Args:
            image_path: Path to image file
            detail_level: 'concise' or 'detailed'
            
        Returns:
            Generated caption text
        """
        if self.client is None:
            return self._fallback_caption(image_path)
        
        try:
            img = Image.open(image_path)
            
            prompt = {
                "concise": "Describe this educational diagram in one sentence, focusing on the key concept.",
                "detailed": "Describe this educational diagram in detail, explaining all visible elements, labels, and relationships."
            }.get(detail_level, "Describe this image briefly.")
            
            response = self.client.generate_content([prompt, img])
            caption = response.text.strip()
            
            logger.debug(f"Generated caption for {Path(image_path).name}: {caption[:100]}...")
            return caption
            
        except Exception as e:
            logger.error(f"Caption generation failed for {image_path}: {e}")
            return self._fallback_caption(image_path)
    
    def _fallback_caption(self, image_path: str) -> str:
        """Fallback caption when API unavailable."""
        filename = Path(image_path).stem
        return f"Educational diagram: {filename.replace('_', ' ')}"


class DiagramExtractor:
    """
    Extract diagrams and figures from PDF documents.
    
    Saves images to disk and returns metadata for evidence bundles.
    """
    
    def __init__(self, output_dir: str = "data/interim/figures"):
        """
        Initialize diagram extractor.
        
        Args:
            output_dir: Directory to save extracted images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_from_pdf(self, pdf_path: str, page_num: Optional[int] = None) -> List[Dict]:
        """
        Extract images from PDF.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Extract from specific page only (None = all pages)
            
        Returns:
            List of dicts with {image_path, page, bbox, caption?}
        """
        import fitz  # PyMuPDF
        
        extracted = []
        doc = fitz.open(pdf_path)
        
        pages = [page_num] if page_num is not None else range(len(doc))
        
        for page_idx in pages:
            page = doc[page_idx]
            images = page.get_images()
            
            for img_idx, img_ref in enumerate(images):
                try:
                    xref = img_ref[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save image
                    pdf_stem = Path(pdf_path).stem
                    img_filename = f"{pdf_stem}_p{page_idx:03d}_img{img_idx:02d}.png"
                    img_path = self.output_dir / img_filename
                    
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Try to find nearby caption (heuristic: text within 50 pts below image)
                    img_bbox = page.get_image_bbox(img_ref)
                    caption = self._find_nearby_text(page, img_bbox)
                    
                    extracted.append({
                        "image_path": str(img_path),
                        "page": page_idx,
                        "bbox": img_bbox,
                        "caption": caption
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_idx} from page {page_idx}: {e}")
        
        doc.close()
        logger.info(f"Extracted {len(extracted)} images from {Path(pdf_path).name}")
        return extracted
    
    def _find_nearby_text(self, page, img_bbox, proximity: float = 50) -> str:
        """Find text near image (likely caption)."""
        try:
            # Search below image
            search_bbox = (img_bbox[0], img_bbox[3], img_bbox[2], img_bbox[3] + proximity)
            text_instances = page.get_text("blocks", clip=search_bbox)
            
            captions = []
            for block in text_instances:
                text = block[4].strip()
                if text and len(text) > 5:  # Filter noise
                    captions.append(text)
            
            return " ".join(captions) if captions else ""
        except:
            return ""


def create_visual_evidence(
    image_path: str,
    caption: str,
    grade: int,
    subject: str,
    chapter: str,
    source_page: int,
    embedding: Optional[np.ndarray] = None
) -> Evidence:
    """
    Create Evidence object for visual content.
    
    Args:
        image_path: Path to image file
        caption: Image caption or description
        grade: Grade level (1-4)
        subject: Subject name
        chapter: Chapter identifier
        source_page: Source page number
        embedding: CLIP embedding (optional)
        
    Returns:
        Evidence object with visual metadata
    """
    evidence = Evidence(
        chunk_id=f"img_{Path(image_path).stem}",
        text=caption,
        grade=grade,
        subject=subject,
        chapter=chapter,
        page=source_page,
        modality="image",  # Mark as image evidence
        metadata={
            "image_path": image_path,
            "has_embedding": embedding is not None,
            "embedding_dim": len(embedding) if embedding is not None else 0
        }
    )
    
    # Store embedding if provided
    if embedding is not None:
        evidence.metadata["clip_embedding"] = embedding.tolist()
    
    return evidence
