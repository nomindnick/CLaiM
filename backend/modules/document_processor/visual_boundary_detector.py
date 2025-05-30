"""Visual boundary detection using embeddings and layout analysis.

This module implements AI-based boundary detection using visual similarity
and layout understanding for improved accuracy with scanned documents.
"""

import io
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import diskcache as dc

from .construction_patterns import detect_document_type, is_strong_document_start


@dataclass
class PageFeatures:
    """Features extracted from a page for boundary detection."""
    page_num: int
    visual_embedding: Optional[np.ndarray] = None
    text_density: float = 0.0
    whitespace_ratio: float = 0.0
    has_letterhead: bool = False
    has_forms: bool = False
    has_drawings: bool = False
    ocr_confidence: float = 1.0
    document_types: List[str] = None
    layout_blocks: int = 0
    orientation: str = "portrait"
    
    def __post_init__(self):
        if self.document_types is None:
            self.document_types = []


@dataclass
class BoundaryScore:
    """Boundary detection score with component breakdown."""
    page_num: int
    total_score: float
    visual_similarity_score: float = 0.0
    layout_change_score: float = 0.0
    pattern_match_score: float = 0.0
    special_features_score: float = 0.0
    confidence: float = 0.0
    reasons: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
    
    @property
    def is_boundary(self) -> bool:
        """Determine if this score indicates a boundary."""
        return self.total_score > 0.6  # Threshold can be tuned


class VisualBoundaryDetector:
    """Hybrid boundary detection using visual and textual features."""
    
    def __init__(self, 
                 cache_dir: str = ".boundary_cache",
                 model_name: str = "clip-ViT-B-32",
                 device: str = None):
        """Initialize the visual boundary detector.
        
        Args:
            cache_dir: Directory for caching embeddings
            model_name: Sentence transformer model to use
            device: Device for model inference (cuda/cpu)
        """
        # Initialize model
        logger.info(f"Loading visual model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Initialize cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = dc.Cache(str(self.cache_dir))
        
        # Configuration
        self.similarity_threshold = 0.7
        self.batch_size = 16
        self.target_dpi = 150  # Lower DPI for faster processing
        
        # Weights for scoring
        self.weights = {
            'visual_similarity': 0.4,
            'layout_change': 0.3,
            'pattern_match': 0.2,
            'special_features': 0.1
        }
    
    def detect_boundaries(self, 
                         pdf_doc: fitz.Document,
                         ocr_handler: Optional[Any] = None,
                         use_cache: bool = True) -> List[BoundaryScore]:
        """Detect document boundaries using visual and textual analysis.
        
        Args:
            pdf_doc: PyMuPDF document
            ocr_handler: OCR handler for scanned pages
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of boundary scores for each page
        """
        logger.info(f"Starting visual boundary detection for {pdf_doc.page_count} pages")
        start_time = time.time()
        
        # Extract features for all pages
        page_features = self._extract_all_features(pdf_doc, ocr_handler, use_cache)
        
        # Compute boundary scores
        boundary_scores = self._compute_boundary_scores(page_features)
        
        # Post-process scores
        boundary_scores = self._post_process_scores(boundary_scores, page_features)
        
        elapsed = time.time() - start_time
        logger.info(f"Visual boundary detection completed in {elapsed:.2f}s")
        
        return boundary_scores
    
    def _extract_all_features(self, 
                             pdf_doc: fitz.Document,
                             ocr_handler: Optional[Any],
                             use_cache: bool) -> List[PageFeatures]:
        """Extract features from all pages."""
        features = []
        
        # Process pages in batches for efficiency
        for batch_start in range(0, pdf_doc.page_count, self.batch_size):
            batch_end = min(batch_start + self.batch_size, pdf_doc.page_count)
            batch_features = self._process_page_batch(
                pdf_doc, batch_start, batch_end, ocr_handler, use_cache
            )
            features.extend(batch_features)
        
        return features
    
    def _process_page_batch(self,
                           pdf_doc: fitz.Document,
                           start_idx: int,
                           end_idx: int,
                           ocr_handler: Optional[Any],
                           use_cache: bool) -> List[PageFeatures]:
        """Process a batch of pages."""
        batch_features = []
        images_to_embed = []
        embed_indices = []
        
        for page_num in range(start_idx, end_idx):
            page = pdf_doc[page_num]
            
            # Extract basic features
            features = self._extract_page_features(page, page_num, ocr_handler)
            
            # Check cache for visual embedding
            cache_key = self._get_cache_key(pdf_doc.name, page_num)
            if use_cache and cache_key in self.cache:
                features.visual_embedding = self.cache[cache_key]
                logger.debug(f"Using cached embedding for page {page_num + 1}")
            else:
                # Convert page to image for batch processing
                try:
                    pix = page.get_pixmap(dpi=self.target_dpi)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    images_to_embed.append(img)
                    embed_indices.append(len(batch_features))
                except Exception as e:
                    logger.error(f"Failed to convert page {page_num + 1} to image: {e}")
            
            batch_features.append(features)
        
        # Batch encode images
        if images_to_embed:
            try:
                embeddings = self.model.encode(images_to_embed, batch_size=self.batch_size)
                for i, embed_idx in enumerate(embed_indices):
                    batch_features[embed_idx].visual_embedding = embeddings[i]
                    # Cache the embedding
                    if use_cache:
                        page_num = batch_features[embed_idx].page_num
                        cache_key = self._get_cache_key(pdf_doc.name, page_num)
                        self.cache[cache_key] = embeddings[i]
            except Exception as e:
                logger.error(f"Failed to encode images: {e}")
        
        return batch_features
    
    def _extract_page_features(self,
                              page: fitz.Page,
                              page_num: int,
                              ocr_handler: Optional[Any]) -> PageFeatures:
        """Extract features from a single page."""
        features = PageFeatures(page_num=page_num)
        
        # Get text (with OCR if needed)
        text = page.get_text()
        ocr_confidence = 1.0
        
        if len(text.strip()) < 10 and ocr_handler:
            try:
                text, ocr_confidence = ocr_handler.process_page(page, dpi=200)
                features.ocr_confidence = ocr_confidence
            except Exception as e:
                logger.debug(f"OCR failed for page {page_num + 1}: {e}")
        
        # Text density
        features.text_density = len(text.strip())
        
        # Whitespace ratio
        features.whitespace_ratio = self._calculate_whitespace_ratio(page)
        
        # Layout analysis
        blocks = page.get_text("blocks")
        features.layout_blocks = len(blocks)
        
        # Document type detection
        if text:
            features.document_types = detect_document_type(text)
        
        # Special features
        features.has_letterhead = self._detect_letterhead(page)
        features.has_forms = self._detect_form_structure(blocks)
        features.has_drawings = self._detect_drawings(page)
        
        # Orientation
        features.orientation = "portrait" if page.rect.height > page.rect.width else "landscape"
        
        return features
    
    def _compute_boundary_scores(self, features: List[PageFeatures]) -> List[BoundaryScore]:
        """Compute boundary scores for each page."""
        scores = []
        
        for i in range(len(features)):
            if i == 0:
                # First page is always a boundary
                score = BoundaryScore(
                    page_num=i,
                    total_score=1.0,
                    confidence=1.0,
                    reasons=["First page"]
                )
            else:
                score = self._compute_page_boundary_score(
                    features[i-1], features[i], i
                )
            
            scores.append(score)
        
        return scores
    
    def _compute_page_boundary_score(self,
                                    prev_features: PageFeatures,
                                    curr_features: PageFeatures,
                                    page_num: int) -> BoundaryScore:
        """Compute boundary score between two pages."""
        score = BoundaryScore(page_num=page_num, total_score=0.0)
        
        # Visual similarity
        if (prev_features.visual_embedding is not None and 
            curr_features.visual_embedding is not None):
            similarity = cosine_similarity(
                [prev_features.visual_embedding],
                [curr_features.visual_embedding]
            )[0][0]
            score.visual_similarity_score = 1.0 - similarity
            if similarity < self.similarity_threshold:
                score.reasons.append(f"Low visual similarity: {similarity:.2f}")
        
        # Layout change
        score.layout_change_score = self._compute_layout_change_score(
            prev_features, curr_features
        )
        
        # Pattern match
        if curr_features.document_types:
            score.pattern_match_score = 1.0
            score.reasons.append(f"Document types: {', '.join(curr_features.document_types)}")
        
        # Special features
        score.special_features_score = self._compute_special_features_score(
            prev_features, curr_features
        )
        
        # Compute total score
        score.total_score = (
            self.weights['visual_similarity'] * score.visual_similarity_score +
            self.weights['layout_change'] * score.layout_change_score +
            self.weights['pattern_match'] * score.pattern_match_score +
            self.weights['special_features'] * score.special_features_score
        )
        
        # Compute confidence based on available features
        available_scores = sum(1 for s in [
            score.visual_similarity_score,
            score.layout_change_score,
            score.pattern_match_score,
            score.special_features_score
        ] if s > 0)
        score.confidence = available_scores / 4.0
        
        return score
    
    def _compute_layout_change_score(self,
                                    prev: PageFeatures,
                                    curr: PageFeatures) -> float:
        """Compute layout change score."""
        score = 0.0
        reasons = []
        
        # Text density change
        if prev.text_density > 0:
            density_ratio = curr.text_density / prev.text_density
            if density_ratio < 0.2 or density_ratio > 5.0:
                score += 0.3
                reasons.append("Significant text density change")
        
        # Whitespace ratio change
        whitespace_diff = abs(curr.whitespace_ratio - prev.whitespace_ratio)
        if whitespace_diff > 0.3:
            score += 0.2
            reasons.append("Whitespace pattern change")
        
        # Layout blocks change
        if prev.layout_blocks > 0:
            block_ratio = curr.layout_blocks / prev.layout_blocks
            if block_ratio < 0.5 or block_ratio > 2.0:
                score += 0.3
                reasons.append("Layout structure change")
        
        # Orientation change
        if prev.orientation != curr.orientation:
            score += 0.2
            reasons.append("Page orientation change")
        
        return min(score, 1.0)
    
    def _compute_special_features_score(self,
                                       prev: PageFeatures,
                                       curr: PageFeatures) -> float:
        """Compute special features score."""
        score = 0.0
        
        # Letterhead appearance
        if not prev.has_letterhead and curr.has_letterhead:
            score += 0.4
        
        # Form transition
        if not prev.has_forms and curr.has_forms:
            score += 0.3
        
        # Drawing transition
        if not prev.has_drawings and curr.has_drawings:
            score += 0.3
        
        return min(score, 1.0)
    
    def _post_process_scores(self,
                            scores: List[BoundaryScore],
                            features: List[PageFeatures]) -> List[BoundaryScore]:
        """Post-process scores to handle special cases."""
        # Smooth scores to reduce noise
        for i in range(1, len(scores) - 1):
            if scores[i].total_score > 0.4 and scores[i].total_score < 0.6:
                # Check context
                if scores[i-1].total_score < 0.3 and scores[i+1].total_score < 0.3:
                    # Isolated spike, reduce confidence
                    scores[i].confidence *= 0.7
                    scores[i].reasons.append("Isolated boundary - reduced confidence")
        
        # Boost scores for construction-specific patterns
        for i, (score, feature) in enumerate(zip(scores, features)):
            if feature.document_types:
                # Strong document types get a boost
                if any(dt in ['EMAIL', 'SUBMITTAL', 'RFI', 'CHANGE_ORDER'] 
                       for dt in feature.document_types):
                    score.total_score = min(score.total_score * 1.2, 1.0)
                    score.reasons.append("Construction document type boost")
        
        return scores
    
    def _calculate_whitespace_ratio(self, page: fitz.Page) -> float:
        """Calculate the ratio of whitespace on the page."""
        try:
            # Get page as image
            pix = page.get_pixmap(dpi=72)  # Low DPI for speed
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            # Convert to grayscale if needed
            if pix.n > 1:
                gray = np.mean(img_array[:,:,:3], axis=2)
            else:
                gray = img_array
            
            # Count white pixels (threshold at 240)
            white_pixels = np.sum(gray > 240)
            total_pixels = gray.size
            
            return white_pixels / total_pixels
        except Exception as e:
            logger.debug(f"Failed to calculate whitespace ratio: {e}")
            return 0.5
    
    def _detect_letterhead(self, page: fitz.Page) -> bool:
        """Detect if page has a letterhead."""
        try:
            # Check for images in top 20% of page
            page_rect = page.rect
            top_region = fitz.Rect(
                page_rect.x0,
                page_rect.y0,
                page_rect.x1,
                page_rect.y0 + (page_rect.height * 0.2)
            )
            
            # Check for images
            images = page.get_images()
            for img in images:
                try:
                    img_rects = page.get_image_bbox(img)
                    for rect in img_rects:
                        if top_region.intersects(rect):
                            return True
                except:
                    pass
            
            # Check for heavy text/graphics in header
            drawings = page.get_drawings()
            header_drawings = sum(1 for d in drawings 
                                if d['rect'].y0 < page_rect.height * 0.2)
            
            return header_drawings > 10
        except Exception as e:
            logger.debug(f"Failed to detect letterhead: {e}")
            return False
    
    def _detect_form_structure(self, blocks: List) -> bool:
        """Detect if page has form-like structure."""
        if len(blocks) < 10:
            return False
        
        # Check for grid-like arrangement
        y_positions = [b[1] for b in blocks]  # y0 coordinates
        unique_y = len(set(y_positions))
        
        # Many aligned blocks suggest a form
        return unique_y > 15 and len(blocks) > 20
    
    def _detect_drawings(self, page: fitz.Page) -> bool:
        """Detect if page has technical drawings."""
        try:
            drawings = page.get_drawings()
            # Many drawing elements suggest technical content
            return len(drawings) > 50
        except:
            return False
    
    def _get_cache_key(self, pdf_name: str, page_num: int) -> str:
        """Generate cache key for embeddings."""
        return f"{pdf_name}_{page_num}"
    
    def convert_to_boundaries(self, 
                             scores: List[BoundaryScore],
                             min_confidence: float = 0.5) -> List[Tuple[int, int]]:
        """Convert boundary scores to document boundaries.
        
        Args:
            scores: List of boundary scores
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (start_page, end_page) tuples
        """
        boundaries = []
        current_start = 0
        
        for score in scores[1:]:  # Skip first page
            if score.is_boundary and score.confidence >= min_confidence:
                boundaries.append((current_start, score.page_num - 1))
                current_start = score.page_num
                logger.info(f"Boundary detected at page {score.page_num + 1}: "
                          f"score={score.total_score:.2f}, "
                          f"confidence={score.confidence:.2f}, "
                          f"reasons={', '.join(score.reasons)}")
        
        # Add last document
        if current_start < len(scores):
            boundaries.append((current_start, len(scores) - 1))
        
        return boundaries