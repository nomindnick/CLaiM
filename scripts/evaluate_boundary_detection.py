#!/usr/bin/env python3
"""Evaluate boundary detection methods on construction documents."""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.modules.document_processor.evaluation import (
    BoundaryEvaluator, create_sample_annotations
)
from loguru import logger


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate boundary detection methods")
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Path to annotations JSON file"
    )
    parser.add_argument(
        "--pdfs",
        nargs="+",
        type=Path,
        help="PDF files to evaluate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("boundary_evaluation_results.json"),
        help="Output path for results"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample annotations file"
    )
    
    args = parser.parse_args()
    
    # Create sample annotations if requested
    if args.create_sample:
        annotations_path = create_sample_annotations()
        logger.info(f"Sample annotations created at {annotations_path}")
        logger.info("Edit this file with actual boundary annotations for your PDFs")
        return
    
    # Load annotations
    if not args.annotations:
        # Try default location
        default_path = Path("tests/test_data/boundary_annotations.json")
        if default_path.exists():
            args.annotations = default_path
        else:
            logger.error("No annotations file specified. Use --create-sample to create one.")
            return
    
    # Initialize evaluator
    evaluator = BoundaryEvaluator(args.annotations)
    
    # Get PDFs to evaluate
    if args.pdfs:
        pdf_paths = args.pdfs
    else:
        # Use PDFs from annotations
        pdf_paths = [Path(pdf) for pdf in evaluator.annotations.keys()]
    
    # Filter existing PDFs
    existing_pdfs = [p for p in pdf_paths if p.exists()]
    if not existing_pdfs:
        logger.error("No valid PDF files found")
        return
    
    logger.info(f"Evaluating {len(existing_pdfs)} PDFs")
    
    # Run comparison
    results = evaluator.compare_methods(existing_pdfs)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary
    evaluator.print_comparison_summary(results)
    
    # Print detailed results for each PDF
    logger.info("\nDetailed Results by PDF:")
    for method, method_results in results.items():
        if method == 'aggregate':
            continue
        logger.info(f"\n{method.upper()}:")
        for pdf_path, metrics in method_results.items():
            if pdf_path == 'aggregate':
                continue
            logger.info(f"  {Path(pdf_path).name}:")
            logger.info(f"    Precision: {metrics.precision:.3f}")
            logger.info(f"    Recall: {metrics.recall:.3f}")
            logger.info(f"    F1 Score: {metrics.f1_score:.3f}")
            logger.info(f"    Exact matches: {metrics.exact_matches}")
            logger.info(f"    Processing time: {metrics.processing_time:.2f}s")


if __name__ == "__main__":
    main()