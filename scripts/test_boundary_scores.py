#!/usr/bin/env python3
"""Test script to analyze boundary detection scores for debugging."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))

import fitz
from modules.document_processor.visual_boundary_detector import VisualBoundaryDetector
from modules.document_processor.boundary_detector import BoundaryDetector
from modules.document_processor.pdf_splitter import PDFSplitter
from loguru import logger

def analyze_visual_scores(pdf_path: str):
    """Analyze visual boundary detection scores."""
    logger.info(f"Analyzing visual scores for: {pdf_path}")
    
    # Open PDF
    pdf_doc = fitz.open(pdf_path)
    logger.info(f"PDF has {pdf_doc.page_count} pages")
    
    # Create visual detector
    detector = VisualBoundaryDetector()
    
    # Get boundary scores
    scores = detector.detect_boundaries(pdf_doc)
    
    # Analyze scores
    logger.info("\nBoundary Scores Analysis:")
    logger.info("="*80)
    
    boundaries_found = []
    
    for i, score in enumerate(scores):
        if i == 0:
            continue  # Skip first page
            
        logger.info(f"\nPage {score.page_num + 1}:")
        logger.info(f"  Total Score: {score.total_score:.3f} {'<-- BOUNDARY' if score.is_boundary else ''}")
        logger.info(f"  Visual Similarity: {score.visual_similarity_score:.3f}")
        logger.info(f"  Layout Change: {score.layout_change_score:.3f}")
        logger.info(f"  Pattern Match: {score.pattern_match_score:.3f}")
        logger.info(f"  Special Features: {score.special_features_score:.3f}")
        logger.info(f"  Confidence: {score.confidence:.3f}")
        
        if score.reasons:
            logger.info(f"  Reasons: {'; '.join(score.reasons)}")
            
        if score.is_boundary:
            boundaries_found.append(score.page_num)
    
    # Convert to boundaries
    boundaries = detector.convert_to_boundaries(scores)
    logger.info(f"\nFinal boundaries: {boundaries}")
    logger.info(f"Total documents found: {len(boundaries)}")
    
    # Close PDF
    pdf_doc.close()
    
    return boundaries, scores

def test_different_thresholds(pdf_path: str):
    """Test with different threshold values."""
    logger.info("\nTesting different is_boundary thresholds:")
    logger.info("="*80)
    
    pdf_doc = fitz.open(pdf_path)
    detector = VisualBoundaryDetector()
    scores = detector.detect_boundaries(pdf_doc)
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in thresholds:
        # Count boundaries at this threshold
        boundary_count = sum(1 for s in scores[1:] if s.total_score > threshold)
        logger.info(f"Threshold {threshold}: {boundary_count} boundaries")
        
        # Show which pages would be boundaries
        boundary_pages = [s.page_num + 1 for s in scores[1:] if s.total_score > threshold]
        if boundary_pages:
            logger.info(f"  Pages: {boundary_pages}")
    
    pdf_doc.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_boundary_scores.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Analyze visual scores
    boundaries, scores = analyze_visual_scores(pdf_path)
    
    # Test different thresholds
    test_different_thresholds(pdf_path)