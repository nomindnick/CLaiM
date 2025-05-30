"""Evaluation framework for boundary detection accuracy.

This module provides tools to evaluate and compare different boundary
detection methods on construction documents.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np

import fitz
from loguru import logger

from .boundary_detector import BoundaryDetector
from .hybrid_boundary_detector import HybridBoundaryDetector, DetectionLevel
from .ocr_handler import OCRHandler


@dataclass
class BoundaryAnnotation:
    """Ground truth annotation for document boundaries."""
    pdf_path: str
    boundaries: List[Tuple[int, int]]  # List of (start, end) page numbers
    document_types: Optional[Dict[int, str]] = None  # Page -> document type
    notes: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Metrics for boundary detection evaluation."""
    precision: float
    recall: float
    f1_score: float
    exact_matches: int
    partial_matches: int
    false_positives: int
    false_negatives: int
    iou_scores: List[float]  # Intersection over Union for each predicted boundary
    average_iou: float
    processing_time: float
    detection_level: str


class BoundaryEvaluator:
    """Evaluate boundary detection performance."""
    
    def __init__(self, annotations_path: Optional[Path] = None):
        """Initialize evaluator.
        
        Args:
            annotations_path: Path to JSON file with ground truth annotations
        """
        self.annotations = {}
        if annotations_path and annotations_path.exists():
            self.load_annotations(annotations_path)
    
    def load_annotations(self, path: Path):
        """Load ground truth annotations from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for ann_data in data:
            # Convert boundary lists to tuples
            if 'boundaries' in ann_data:
                ann_data['boundaries'] = [tuple(b) for b in ann_data['boundaries']]
            annotation = BoundaryAnnotation(**ann_data)
            self.annotations[annotation.pdf_path] = annotation
        
        logger.info(f"Loaded annotations for {len(self.annotations)} PDFs")
    
    def add_annotation(self, pdf_path: str, boundaries: List[Tuple[int, int]]):
        """Add a single annotation."""
        self.annotations[pdf_path] = BoundaryAnnotation(
            pdf_path=pdf_path,
            boundaries=boundaries
        )
    
    def evaluate_detector(self,
                         detector,
                         pdf_paths: List[Path],
                         detection_level: Optional[DetectionLevel] = None) -> Dict[str, EvaluationMetrics]:
        """Evaluate a detector on multiple PDFs.
        
        Args:
            detector: Boundary detector instance
            pdf_paths: List of PDF files to evaluate
            detection_level: Detection level for hybrid detector
            
        Returns:
            Dictionary mapping PDF paths to metrics
        """
        results = {}
        
        for pdf_path in pdf_paths:
            if str(pdf_path) not in self.annotations:
                logger.warning(f"No annotations for {pdf_path}, skipping")
                continue
            
            logger.info(f"Evaluating {pdf_path.name}")
            metrics = self.evaluate_single_pdf(detector, pdf_path, detection_level)
            results[str(pdf_path)] = metrics
        
        # Compute aggregate metrics
        aggregate = self._compute_aggregate_metrics(results)
        results['aggregate'] = aggregate
        
        return results
    
    def evaluate_single_pdf(self,
                           detector,
                           pdf_path: Path,
                           detection_level: Optional[DetectionLevel] = None) -> EvaluationMetrics:
        """Evaluate detector on a single PDF."""
        import time
        
        # Get ground truth
        ground_truth = self.annotations[str(pdf_path)]
        
        # Run detection
        start_time = time.time()
        pdf_doc = fitz.open(str(pdf_path))
        
        if hasattr(detector, 'detect_boundaries') and detection_level:
            # Hybrid detector
            result = detector.detect_boundaries(pdf_doc, max_level=detection_level)
            predicted = result.boundaries
            level_used = result.detection_level.name.lower()
        else:
            # Simple detector
            predicted = detector.detect_boundaries(pdf_doc)
            level_used = "pattern"
        
        processing_time = time.time() - start_time
        pdf_doc.close()
        
        # Compute metrics
        metrics = self._compute_metrics(
            ground_truth.boundaries,
            predicted,
            processing_time,
            level_used
        )
        
        return metrics
    
    def _compute_metrics(self,
                        ground_truth: List[Tuple[int, int]],
                        predicted: List[Tuple[int, int]],
                        processing_time: float,
                        detection_level: str) -> EvaluationMetrics:
        """Compute evaluation metrics."""
        # Convert to sets for comparison (ensure tuples)
        gt_set = set(tuple(b) if isinstance(b, list) else b for b in ground_truth)
        pred_set = set(tuple(b) if isinstance(b, list) else b for b in predicted)
        
        # Exact matches
        exact_matches = len(gt_set & pred_set)
        
        # False positives and negatives
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)
        
        # Compute IoU scores for partial matches
        iou_scores = []
        partial_matches = 0
        
        for pred_start, pred_end in predicted:
            best_iou = 0.0
            for gt_start, gt_end in ground_truth:
                # Compute IoU
                intersection_start = max(pred_start, gt_start)
                intersection_end = min(pred_end, gt_end)
                
                if intersection_start <= intersection_end:
                    intersection = intersection_end - intersection_start + 1
                    union = (pred_end - pred_start + 1) + (gt_end - gt_start + 1) - intersection
                    iou = intersection / union
                    best_iou = max(best_iou, iou)
                    
                    if iou > 0.5:  # Partial match threshold
                        partial_matches += 1
            
            iou_scores.append(best_iou)
        
        # Compute precision, recall, F1
        precision = exact_matches / len(predicted) if predicted else 0.0
        recall = exact_matches / len(ground_truth) if ground_truth else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Average IoU
        average_iou = np.mean(iou_scores) if iou_scores else 0.0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            exact_matches=exact_matches,
            partial_matches=partial_matches,
            false_positives=false_positives,
            false_negatives=false_negatives,
            iou_scores=iou_scores,
            average_iou=average_iou,
            processing_time=processing_time,
            detection_level=detection_level
        )
    
    def _compute_aggregate_metrics(self, results: Dict[str, EvaluationMetrics]) -> EvaluationMetrics:
        """Compute aggregate metrics across multiple PDFs."""
        # Filter out non-metric entries
        metrics_list = [m for k, m in results.items() if k != 'aggregate']
        
        if not metrics_list:
            return EvaluationMetrics(
                precision=0, recall=0, f1_score=0,
                exact_matches=0, partial_matches=0,
                false_positives=0, false_negatives=0,
                iou_scores=[], average_iou=0,
                processing_time=0, detection_level="none"
            )
        
        # Aggregate metrics
        total_exact = sum(m.exact_matches for m in metrics_list)
        total_partial = sum(m.partial_matches for m in metrics_list)
        total_fp = sum(m.false_positives for m in metrics_list)
        total_fn = sum(m.false_negatives for m in metrics_list)
        all_ious = [iou for m in metrics_list for iou in m.iou_scores]
        
        # Compute aggregate precision/recall
        total_predicted = total_exact + total_fp
        total_ground_truth = total_exact + total_fn
        
        precision = total_exact / total_predicted if total_predicted > 0 else 0.0
        recall = total_exact / total_ground_truth if total_ground_truth > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            exact_matches=total_exact,
            partial_matches=total_partial,
            false_positives=total_fp,
            false_negatives=total_fn,
            iou_scores=all_ious,
            average_iou=np.mean(all_ious) if all_ious else 0.0,
            processing_time=sum(m.processing_time for m in metrics_list),
            detection_level=metrics_list[0].detection_level if metrics_list else "none"
        )
    
    def compare_methods(self, pdf_paths: List[Path]) -> Dict[str, Dict[str, EvaluationMetrics]]:
        """Compare different detection methods."""
        results = {}
        
        # Initialize detectors
        ocr_handler = OCRHandler()
        pattern_detector = BoundaryDetector(ocr_handler)
        hybrid_detector = HybridBoundaryDetector(ocr_handler)
        
        # Evaluate pattern-based detection
        logger.info("Evaluating pattern-based detection...")
        results['pattern'] = self.evaluate_detector(pattern_detector, pdf_paths)
        
        # Evaluate hybrid with visual detection
        logger.info("Evaluating hybrid detection with visual analysis...")
        results['hybrid_visual'] = self.evaluate_detector(
            hybrid_detector, pdf_paths, DetectionLevel.VISUAL
        )
        
        # Evaluate hybrid with deep detection (if available)
        logger.info("Evaluating hybrid detection with deep analysis...")
        results['hybrid_deep'] = self.evaluate_detector(
            hybrid_detector, pdf_paths, DetectionLevel.DEEP
        )
        
        return results
    
    def save_results(self, results: Dict, output_path: Path):
        """Save evaluation results to JSON."""
        # Convert to serializable format
        serializable = {}
        for method, method_results in results.items():
            serializable[method] = {}
            for pdf_path, metrics in method_results.items():
                serializable[method][pdf_path] = asdict(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
    
    def print_comparison_summary(self, results: Dict):
        """Print a summary comparison of methods."""
        logger.info("\n" + "="*60)
        logger.info("BOUNDARY DETECTION EVALUATION SUMMARY")
        logger.info("="*60)
        
        # Headers
        headers = ["Method", "Precision", "Recall", "F1", "Exact", "Partial", "Avg IoU", "Time(s)"]
        col_widths = [15, 10, 10, 10, 8, 8, 10, 10]
        
        # Print header
        header_line = ""
        for header, width in zip(headers, col_widths):
            header_line += f"{header:<{width}}"
        logger.info(header_line)
        logger.info("-" * sum(col_widths))
        
        # Print results for each method
        for method, method_results in results.items():
            if 'aggregate' in method_results:
                metrics = method_results['aggregate']
                row = [
                    method,
                    f"{metrics.precision:.3f}",
                    f"{metrics.recall:.3f}",
                    f"{metrics.f1_score:.3f}",
                    str(metrics.exact_matches),
                    str(metrics.partial_matches),
                    f"{metrics.average_iou:.3f}",
                    f"{metrics.processing_time:.2f}"
                ]
                
                row_line = ""
                for val, width in zip(row, col_widths):
                    row_line += f"{val:<{width}}"
                logger.info(row_line)
        
        logger.info("="*60)


def create_sample_annotations():
    """Create sample annotations for testing."""
    annotations = [
        {
            "pdf_path": "tests/test_data/Mixed_Document_Contract_Amendment.pdf",
            "boundaries": [(0, 2), (3, 5), (6, 8)],  # Example boundaries
            "document_types": {
                0: "CONTRACT",
                3: "EMAIL",
                6: "INVOICE"
            },
            "notes": "Mixed document with contract, emails, and invoice"
        },
        {
            "pdf_path": "tests/Sample PDFs.pdf",
            "boundaries": [(0, 0), (1, 2), (3, 4), (5, 6)],  # Example boundaries
            "notes": "Sample construction documents"
        }
    ]
    
    output_path = Path("tests/test_data/boundary_annotations.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    logger.info(f"Created sample annotations at {output_path}")
    return output_path