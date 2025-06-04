#!/usr/bin/env python3
"""Test script for LLM-based boundary detection."""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import fitz
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.modules.document_processor.llm_boundary_detector import (
    LLMBoundaryDetector, BoundaryAnalysis, WindowAnalysis, ConfidenceLevel
)
from backend.modules.document_processor.hybrid_boundary_detector import HybridBoundaryDetector
from backend.modules.document_processor.improved_ocr_handler import ImprovedOCRHandler
from backend.modules.llm_client.ollama_client import OllamaClient

console = Console()


def load_ground_truth(pdf_path: Path) -> List[Dict]:
    """Load ground truth annotations for a PDF."""
    # Check for ground truth file
    ground_truth_file = pdf_path.parent / "Test_PDF_Set_Ground_Truth.json"
    if ground_truth_file.exists():
        with open(ground_truth_file, 'r') as f:
            data = json.load(f)
            filename = pdf_path.name
            if filename in data:
                return data[filename]
    return []


def test_window_extraction():
    """Test sliding window extraction functionality."""
    console.print("\n[bold cyan]Testing Window Extraction[/bold cyan]")
    
    # Create test PDF with known content
    pdf_path = Path("tests/test_data/Mixed_Document_Contract_Amendment.pdf")
    if not pdf_path.exists():
        console.print(f"[red]Test PDF not found: {pdf_path}[/red]")
        return
    
    pdf_doc = fitz.open(pdf_path)
    detector = LLMBoundaryDetector(window_size=3, overlap=1)
    
    # Extract windows
    windows = detector._extract_page_windows(pdf_doc)
    
    console.print(f"PDF has {pdf_doc.page_count} pages")
    console.print(f"Created {len(windows)} windows with size=3, overlap=1")
    
    # Display window details
    table = Table(title="Window Analysis")
    table.add_column("Window", style="cyan")
    table.add_column("Pages", style="green")
    table.add_column("Text Preview", style="yellow")
    
    for i, window in enumerate(windows[:5]):  # Show first 5 windows
        pages = [p['page_num'] + 1 for p in window]
        page_range = f"{pages[0]}-{pages[-1]}"
        
        # Get text preview from first page in window
        text_preview = window[0]['text'][:50].replace('\n', ' ')
        if len(window[0]['text']) > 50:
            text_preview += "..."
            
        table.add_row(f"Window {i+1}", page_range, text_preview)
    
    console.print(table)
    pdf_doc.close()


def test_llm_analysis():
    """Test LLM analysis on a single window."""
    console.print("\n[bold cyan]Testing LLM Analysis[/bold cyan]")
    
    pdf_path = Path("tests/test_data/Mixed_Document_Contract_Amendment.pdf")
    if not pdf_path.exists():
        console.print(f"[red]Test PDF not found: {pdf_path}[/red]")
        return
    
    pdf_doc = fitz.open(pdf_path)
    
    # Initialize detector with Ollama
    llm_client = OllamaClient(model="llama3:8b-instruct-q4_0")
    detector = LLMBoundaryDetector(llm_client=llm_client)
    
    # Get first window
    windows = detector._extract_page_windows(pdf_doc)
    if not windows:
        console.print("[red]No windows extracted[/red]")
        return
    
    # Analyze first window
    console.print(f"\nAnalyzing first window (pages 1-{len(windows[0])})")
    start_time = time.time()
    
    analysis = detector._analyze_window(windows[0])
    
    elapsed = time.time() - start_time
    console.print(f"Analysis took {elapsed:.2f} seconds")
    
    # Display results
    console.print(f"\nWindow Summary: {analysis.window_summary}")
    console.print(f"Average Confidence: {analysis.avg_confidence:.2f}")
    
    table = Table(title="Boundary Analysis")
    table.add_column("Page", style="cyan")
    table.add_column("Is Boundary", style="green")
    table.add_column("Confidence", style="yellow")
    table.add_column("Type", style="magenta")
    table.add_column("Reasoning", style="white")
    
    for boundary in analysis.boundaries:
        table.add_row(
            str(boundary.page_num + 1),
            "Yes" if boundary.is_boundary else "No",
            f"{boundary.confidence:.2f}",
            boundary.document_type or "Unknown",
            boundary.reasoning[:50] + "..." if len(boundary.reasoning) > 50 else boundary.reasoning
        )
    
    console.print(table)
    pdf_doc.close()


def test_full_detection():
    """Test full boundary detection on a PDF."""
    console.print("\n[bold cyan]Testing Full Boundary Detection[/bold cyan]")
    
    pdf_path = Path("tests/test_data/Mixed_Document_Contract_Amendment.pdf")
    if not pdf_path.exists():
        console.print(f"[red]Test PDF not found: {pdf_path}[/red]")
        return
    
    pdf_doc = fitz.open(pdf_path)
    
    # Initialize detector
    llm_client = OllamaClient(model="llama3:8b-instruct-q4_0")
    detector = LLMBoundaryDetector(
        llm_client=llm_client,
        window_size=3,
        overlap=1,
        confidence_threshold=0.7
    )
    
    console.print(f"Processing {pdf_doc.page_count} pages...")
    start_time = time.time()
    
    # Detect boundaries
    boundaries = detector.detect_boundaries(pdf_doc)
    
    elapsed = time.time() - start_time
    console.print(f"\nDetection took {elapsed:.2f} seconds")
    console.print(f"Found {len(boundaries)} documents")
    
    # Display results
    table = Table(title="Detected Documents")
    table.add_column("Doc #", style="cyan")
    table.add_column("Pages", style="green")
    table.add_column("Page Count", style="yellow")
    
    for i, (start, end) in enumerate(boundaries):
        table.add_row(
            str(i + 1),
            f"{start + 1}-{end + 1}",
            str(end - start + 1)
        )
    
    console.print(table)
    
    # Load ground truth if available
    ground_truth = load_ground_truth(pdf_path)
    if ground_truth:
        console.print(f"\n[bold]Ground Truth:[/bold] {len(ground_truth)} documents")
        accuracy = calculate_accuracy(boundaries, ground_truth)
        console.print(f"[bold]Accuracy:[/bold] {accuracy:.2%}")
    
    pdf_doc.close()


def compare_detection_methods():
    """Compare LLM vs pattern-based detection."""
    console.print("\n[bold cyan]Comparing Detection Methods[/bold cyan]")
    
    test_pdfs = [
        "tests/test_data/Mixed_Document_Contract_Amendment.pdf",
        "tests/Test_PDF_Set_1.pdf",
        "tests/Test_PDF_Set_2.pdf"
    ]
    
    results = []
    
    for pdf_path_str in test_pdfs:
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            console.print(f"[yellow]Skipping {pdf_path} (not found)[/yellow]")
            continue
        
        console.print(f"\n[bold]Testing {pdf_path.name}[/bold]")
        pdf_doc = fitz.open(pdf_path)
        
        # Pattern-based detection
        console.print("Running pattern-based detection...")
        pattern_detector = HybridBoundaryDetector()
        start_time = time.time()
        pattern_boundaries = pattern_detector.detect_boundaries(pdf_doc)
        pattern_time = time.time() - start_time
        
        # LLM-based detection
        console.print("Running LLM-based detection...")
        llm_client = OllamaClient(model="llama3:8b-instruct-q4_0")
        llm_detector = LLMBoundaryDetector(llm_client=llm_client)
        start_time = time.time()
        llm_boundaries = llm_detector.detect_boundaries(pdf_doc)
        llm_time = time.time() - start_time
        
        # Ground truth
        ground_truth = load_ground_truth(pdf_path)
        
        # Calculate accuracies
        pattern_accuracy = calculate_accuracy(pattern_boundaries, ground_truth) if ground_truth else None
        llm_accuracy = calculate_accuracy(llm_boundaries, ground_truth) if ground_truth else None
        
        results.append({
            'pdf': pdf_path.name,
            'pages': pdf_doc.page_count,
            'pattern_docs': len(pattern_boundaries),
            'llm_docs': len(llm_boundaries),
            'ground_truth': len(ground_truth) if ground_truth else None,
            'pattern_time': pattern_time,
            'llm_time': llm_time,
            'pattern_accuracy': pattern_accuracy,
            'llm_accuracy': llm_accuracy
        })
        
        pdf_doc.close()
    
    # Display comparison table
    table = Table(title="Method Comparison")
    table.add_column("PDF", style="cyan")
    table.add_column("Pages", style="green")
    table.add_column("Pattern Docs", style="yellow")
    table.add_column("LLM Docs", style="yellow")
    table.add_column("Truth", style="magenta")
    table.add_column("Pattern Time", style="red")
    table.add_column("LLM Time", style="red")
    table.add_column("Pattern Acc", style="blue")
    table.add_column("LLM Acc", style="blue")
    
    for result in results:
        table.add_row(
            result['pdf'],
            str(result['pages']),
            str(result['pattern_docs']),
            str(result['llm_docs']),
            str(result['ground_truth']) if result['ground_truth'] else "N/A",
            f"{result['pattern_time']:.2f}s",
            f"{result['llm_time']:.2f}s",
            f"{result['pattern_accuracy']:.2%}" if result['pattern_accuracy'] else "N/A",
            f"{result['llm_accuracy']:.2%}" if result['llm_accuracy'] else "N/A"
        )
    
    console.print(table)


def calculate_accuracy(detected: List[Tuple[int, int]], ground_truth: List[Dict]) -> float:
    """Calculate accuracy of boundary detection."""
    if not ground_truth:
        return 0.0
    
    # Convert ground truth to same format
    truth_boundaries = []
    for doc in ground_truth:
        start = doc['start_page'] - 1  # Convert to 0-based
        end = doc['end_page'] - 1
        truth_boundaries.append((start, end))
    
    # Simple accuracy: correct count / total
    if len(detected) == len(truth_boundaries):
        # Check if boundaries match
        matches = 0
        for det in detected:
            for truth in truth_boundaries:
                if abs(det[0] - truth[0]) <= 1 and abs(det[1] - truth[1]) <= 1:
                    matches += 1
                    break
        return matches / len(truth_boundaries)
    
    # Penalize for wrong count
    return max(0, 1 - abs(len(detected) - len(truth_boundaries)) / len(truth_boundaries))


def test_confidence_levels():
    """Test confidence level analysis."""
    console.print("\n[bold cyan]Testing Confidence Levels[/bold cyan]")
    
    # Create some test boundary analyses
    boundaries = [
        BoundaryAnalysis(0, True, 0.95, "email", "Clear email header", False),
        BoundaryAnalysis(1, False, 0.15, None, "Quoted content", True),
        BoundaryAnalysis(2, False, 0.75, None, "Possible continuation", True),
        BoundaryAnalysis(3, True, 0.65, "invoice", "Invoice header detected", False),
        BoundaryAnalysis(4, False, 0.45, None, "Unclear boundary", True),
    ]
    
    # Display confidence analysis
    table = Table(title="Confidence Level Analysis")
    table.add_column("Page", style="cyan")
    table.add_column("Boundary", style="green")
    table.add_column("Confidence", style="yellow")
    table.add_column("Level", style="magenta")
    table.add_column("Action", style="white")
    
    for b in boundaries:
        level = b.confidence_level
        if level == ConfidenceLevel.HIGH:
            action = "Auto-accept"
            level_str = "HIGH"
        elif level == ConfidenceLevel.MEDIUM:
            action = "Review suggested"
            level_str = "MEDIUM"
        else:
            action = "Manual review"
            level_str = "LOW"
        
        table.add_row(
            str(b.page_num + 1),
            "Yes" if b.is_boundary else "No",
            f"{b.confidence:.2f}",
            level_str,
            action
        )
    
    console.print(table)


def main():
    """Run all tests."""
    console.print("[bold green]LLM Boundary Detection Test Suite[/bold green]")
    
    try:
        # Check if Ollama is available
        client = OllamaClient(model="llama3:8b-instruct-q4_0")
        test_response = client.complete("Hello")
        console.print("[green]✓ Ollama connection successful[/green]")
    except Exception as e:
        console.print(f"[red]✗ Ollama not available: {e}[/red]")
        console.print("[yellow]Please ensure Ollama is running with llama3:8b model[/yellow]")
        return
    
    # Run tests
    test_window_extraction()
    test_llm_analysis()
    test_confidence_levels()
    test_full_detection()
    compare_detection_methods()
    
    console.print("\n[bold green]All tests completed![/bold green]")


if __name__ == "__main__":
    main()