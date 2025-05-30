#!/usr/bin/env python3
"""
Comprehensive pipeline evaluation script for Test_PDF_Set_1.pdf
Compares actual results against ground truth and identifies improvement areas.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.pdf_splitter import PDFSplitter
from modules.document_processor.models import PDFProcessingRequest
from modules.ai_classifier.classifier import DocumentClassifier
from modules.storage.storage_manager import StorageManager
import tempfile

# Ground truth data
GROUND_TRUTH = {
    "documents": [
        {"pages": "1-4", "type": "Email Chain", "summary": "Email exchange from February 26-28, 2024 between BDM Inc (Lyle Bolte) and Integrated Designs/SOMAM regarding RFI responses #3-6"},
        {"pages": "5-6", "type": "Email Chain", "summary": "Email from February 22-28, 2024 from Javier Moreno (Fowler USD) announcing Fowler HS Gym construction schedule"},
        {"pages": "7-8", "type": "Submittal", "summary": "Submittal Transmittal #0007 dated 2/29/2024 for Schedule of Values, marked 'Revise and Resubmit'"},
        {"pages": "9-12", "type": "Schedule of Values", "summary": "BDM Inc's Schedule of Values dated 2/27/2024 showing detailed breakdown of $1,459,395.00 contract"},
        {"pages": "13", "type": "Email", "summary": "Email from Lyle Bolte dated February 29, 2024 resubmitting Schedule of Values"},
        {"pages": "14-17", "type": "Application for Payment", "summary": "Application and Certificate for Payment (AIA G702/G703) for period ending 1/31/2024"},
        {"pages": "18-19", "type": "Invoice", "summary": "MicroMetl packing slips dated 12/15/2023 and 1/31/2024 for Genesis economizers"},
        {"pages": "20-22", "type": "Invoice", "summary": "Geary Pacific Supply sales order packing slips #5410182 and #5410184 dated 2/7/2024"},
        {"pages": "23-25", "type": "Request for Information", "summary": "RFI #7 dated March 28, 2024 from BDM Inc regarding gas line connection points"},
        {"pages": "26-31", "type": "Plans and Specifications", "summary": "Structural engineering drawings (sheets SI-1.1 through SI-1.6) showing HVAC duct support details"},
        {"pages": "32-33", "type": "Cost Proposal", "summary": "Cost Proposal #2 dated 4/5/2024 for $6,707.66 to saw cut, demo and remove existing concrete"},
        {"pages": "34", "type": "Cost Proposal", "summary": "Cost Proposal #3 dated 4/5/2024 for $19,295.51 to remove and install new HP-1 Wall Mount Bard Unit"},
        {"pages": "35", "type": "Cost Proposal", "summary": "Cost Proposal #4 dated 4/5/2024 for $85,694.31 for 6 additional HP-2 Bard Units"},
        {"pages": "36", "type": "Email", "summary": "Email dated April 9, 2024 from Christine Hoskins to May Yang requesting review"}
    ]
}

def parse_page_range(page_range: str) -> tuple:
    """Parse page range string like '1-4' into (start, end) tuple."""
    if '-' in page_range:
        start, end = page_range.split('-')
        return (int(start), int(end))
    else:
        page = int(page_range)
        return (page, page)

def calculate_page_coverage(actual_boundaries: List[tuple], total_pages: int) -> Dict[str, Any]:
    """Calculate how well the detected boundaries cover all pages."""
    covered_pages = set()
    for start, end in actual_boundaries:
        covered_pages.update(range(start, end + 1))
    
    coverage_percentage = len(covered_pages) / total_pages * 100
    missing_pages = set(range(1, total_pages + 1)) - covered_pages
    
    return {
        "coverage_percentage": coverage_percentage,
        "covered_pages": len(covered_pages),
        "total_pages": total_pages,
        "missing_pages": sorted(list(missing_pages))
    }

def compare_boundaries(actual: List[tuple], expected: List[tuple]) -> Dict[str, Any]:
    """Compare actual boundary detection with expected ground truth."""
    # Convert expected page ranges to boundary format
    expected_boundaries = []
    for doc in GROUND_TRUTH["documents"]:
        start, end = parse_page_range(doc["pages"])
        expected_boundaries.append((start, end))
    
    exact_matches = 0
    partial_matches = 0
    missed_documents = 0
    extra_documents = len(actual) - len(expected_boundaries)
    
    for exp_start, exp_end in expected_boundaries:
        found_match = False
        for act_start, act_end in actual:
            if act_start == exp_start and act_end == exp_end:
                exact_matches += 1
                found_match = True
                break
            elif (act_start <= exp_start <= act_end) or (exp_start <= act_start <= exp_end):
                partial_matches += 1
                found_match = True
                break
        
        if not found_match:
            missed_documents += 1
    
    return {
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "missed_documents": missed_documents,
        "extra_documents": max(0, extra_documents),
        "total_expected": len(expected_boundaries),
        "total_actual": len(actual),
        "boundary_accuracy": exact_matches / len(expected_boundaries) * 100 if expected_boundaries else 0
    }

def compare_classifications(actual_docs: List[Dict], expected_docs: List[Dict]) -> Dict[str, Any]:
    """Compare document type classifications."""
    correct_classifications = 0
    total_classifications = len(expected_docs)
    classification_errors = []
    
    # Create mapping of page ranges to expected types
    expected_by_pages = {}
    for doc in expected_docs:
        start, end = parse_page_range(doc["pages"])
        expected_by_pages[(start, end)] = doc["type"]
    
    # Check each actual document against expected
    for doc in actual_docs:
        # Find the closest expected document by page range
        doc_pages = (doc.get("start_page", 0), doc.get("end_page", 0))
        
        best_match = None
        best_overlap = 0
        
        for exp_pages, exp_type in expected_by_pages.items():
            # Calculate overlap
            overlap_start = max(doc_pages[0], exp_pages[0])
            overlap_end = min(doc_pages[1], exp_pages[1])
            overlap = max(0, overlap_end - overlap_start + 1)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = (exp_pages, exp_type)
        
        if best_match and best_overlap > 0:
            actual_type = doc.get("document_type", "Unknown")
            expected_type = best_match[1]
            
            if actual_type == expected_type:
                correct_classifications += 1
            else:
                classification_errors.append({
                    "pages": f"{doc_pages[0]}-{doc_pages[1]}",
                    "expected": expected_type,
                    "actual": actual_type,
                    "overlap": best_overlap
                })
    
    classification_accuracy = correct_classifications / total_classifications * 100 if total_classifications > 0 else 0
    
    return {
        "correct_classifications": correct_classifications,
        "total_expected": total_classifications,
        "classification_accuracy": classification_accuracy,
        "classification_errors": classification_errors
    }

def analyze_pipeline_gaps(boundary_analysis: Dict, classification_analysis: Dict, coverage_analysis: Dict) -> List[str]:
    """Analyze results and identify specific areas for improvement."""
    gaps = []
    
    # Boundary detection gaps
    if coverage_analysis["coverage_percentage"] < 100:
        gaps.append(f"Page Coverage: Only {coverage_analysis['coverage_percentage']:.1f}% of pages covered. Missing pages: {coverage_analysis['missing_pages']}")
    
    if boundary_analysis["boundary_accuracy"] < 80:
        gaps.append(f"Boundary Accuracy: Only {boundary_analysis['boundary_accuracy']:.1f}% exact matches. Need better boundary detection.")
    
    if boundary_analysis["missed_documents"] > 0:
        gaps.append(f"Missed Documents: {boundary_analysis['missed_documents']} documents not detected at all.")
    
    if boundary_analysis["extra_documents"] > 0:
        gaps.append(f"Over-segmentation: {boundary_analysis['extra_documents']} extra document boundaries detected.")
    
    # Classification gaps
    if classification_analysis["classification_accuracy"] < 80:
        gaps.append(f"Classification Accuracy: Only {classification_analysis['classification_accuracy']:.1f}% correct classifications.")
    
    # Specific classification issues
    error_types = {}
    for error in classification_analysis["classification_errors"]:
        expected = error["expected"]
        actual = error["actual"]
        key = f"{expected} → {actual}"
        error_types[key] = error_types.get(key, 0) + 1
    
    for error_pattern, count in error_types.items():
        gaps.append(f"Classification Error: {count} documents misclassified as {error_pattern}")
    
    return gaps

def main():
    """Main test execution."""
    test_pdf_path = "/home/nick/Projects/CLaiM/tests/Test_PDF_Set_1.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"Error: Test PDF not found at {test_pdf_path}")
        return
    
    print("🔍 Testing Document Ingestion Pipeline with Test_PDF_Set_1.pdf")
    print("=" * 70)
    
    # Initialize components
    try:
        pdf_splitter = PDFSplitter()
        classifier = DocumentClassifier()
        
        # Process the PDF
        print("📄 Processing PDF through current pipeline...")
        request = PDFProcessingRequest(
            file_path=test_pdf_path,
            user_id="test_user"
        )
        result = pdf_splitter.process_pdf(request)
        documents = result.documents
        
        print(f"✅ Pipeline extracted {len(documents)} documents")
        
        # Get total page count for coverage analysis
        import fitz
        pdf_doc = fitz.open(test_pdf_path)
        total_pages = pdf_doc.page_count
        pdf_doc.close()
        
        print(f"📊 Total pages in PDF: {total_pages}")
        
        # Extract boundary information
        actual_boundaries = []
        actual_docs_info = []
        
        for doc in documents:
            start_page = doc.page_range[0]
            end_page = doc.page_range[1]
            actual_boundaries.append((start_page, end_page))
            
            actual_docs_info.append({
                "start_page": start_page,
                "end_page": end_page,
                "document_type": doc.type.value,  # Get the enum value
                "title": doc.metadata.title[:100] if doc.metadata.title else "No title",
                "text_preview": doc.text[:200] if doc.text else "No text"
            })
        
        print("\n📋 Detected Documents:")
        for i, doc_info in enumerate(actual_docs_info):
            print(f"{i+1:2d}. Pages {doc_info['start_page']:2d}-{doc_info['end_page']:2d} | {doc_info['document_type']:20s} | {doc_info['title']}")
        
        # Analyze results
        print("\n🔍 Analysis Results:")
        print("-" * 50)
        
        # Page coverage analysis
        coverage_analysis = calculate_page_coverage(actual_boundaries, total_pages)
        print(f"📄 Page Coverage: {coverage_analysis['coverage_percentage']:.1f}% ({coverage_analysis['covered_pages']}/{coverage_analysis['total_pages']} pages)")
        if coverage_analysis['missing_pages']:
            print(f"   Missing pages: {coverage_analysis['missing_pages']}")
        
        # Boundary analysis
        expected_boundaries = [(parse_page_range(doc["pages"])) for doc in GROUND_TRUTH["documents"]]
        boundary_analysis = compare_boundaries(actual_boundaries, expected_boundaries)
        print(f"🎯 Boundary Accuracy: {boundary_analysis['boundary_accuracy']:.1f}% ({boundary_analysis['exact_matches']}/{boundary_analysis['total_expected']} exact matches)")
        print(f"   Partial matches: {boundary_analysis['partial_matches']}")
        print(f"   Missed documents: {boundary_analysis['missed_documents']}")
        print(f"   Extra documents: {boundary_analysis['extra_documents']}")
        
        # Classification analysis
        classification_analysis = compare_classifications(actual_docs_info, GROUND_TRUTH["documents"])
        print(f"🏷️  Classification Accuracy: {classification_analysis['classification_accuracy']:.1f}% ({classification_analysis['correct_classifications']}/{classification_analysis['total_expected']} correct)")
        
        if classification_analysis['classification_errors']:
            print("   Classification errors:")
            for error in classification_analysis['classification_errors'][:5]:  # Show first 5 errors
                print(f"     Pages {error['pages']}: Expected '{error['expected']}', Got '{error['actual']}'")
        
        # Gap analysis
        print(f"\n🚨 Identified Improvement Areas:")
        gaps = analyze_pipeline_gaps(boundary_analysis, classification_analysis, coverage_analysis)
        for i, gap in enumerate(gaps, 1):
            print(f"{i:2d}. {gap}")
        
        # Detailed comparison table
        print(f"\n📊 Detailed Comparison (Expected vs Actual):")
        print("-" * 90)
        print(f"{'Expected':<35} {'Actual':<35} {'Match':<10}")
        print("-" * 90)
        
        for i, expected_doc in enumerate(GROUND_TRUTH["documents"]):
            exp_pages = expected_doc["pages"]
            exp_type = expected_doc["type"]
            
            # Find closest actual document
            exp_start, exp_end = parse_page_range(exp_pages)
            best_actual = None
            best_overlap = 0
            
            for actual_doc in actual_docs_info:
                act_start = actual_doc["start_page"]
                act_end = actual_doc["end_page"]
                
                overlap_start = max(exp_start, act_start)
                overlap_end = min(exp_end, act_end)
                overlap = max(0, overlap_end - overlap_start + 1)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_actual = actual_doc
            
            if best_actual:
                actual_pages = f"{best_actual['start_page']}-{best_actual['end_page']}"
                actual_type = best_actual['document_type']
                match_status = "✅ Yes" if (exp_pages == actual_pages and exp_type == actual_type) else "❌ No"
            else:
                actual_pages = "NOT FOUND"
                actual_type = "NOT FOUND"
                match_status = "❌ No"
            
            expected_str = f"{exp_pages} | {exp_type}"
            actual_str = f"{actual_pages} | {actual_type}"
            print(f"{expected_str:<35} {actual_str:<35} {match_status:<10}")
        
        # Save detailed results for analysis
        results = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "test_file": test_pdf_path,
            "total_pages": total_pages,
            "coverage_analysis": coverage_analysis,
            "boundary_analysis": boundary_analysis,
            "classification_analysis": classification_analysis,
            "improvement_gaps": gaps,
            "actual_documents": actual_docs_info,
            "expected_documents": GROUND_TRUTH["documents"]
        }
        
        results_path = "/home/nick/Projects/CLaiM/test_results_detailed.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_path}")
        
        # Summary score
        overall_score = (coverage_analysis['coverage_percentage'] + 
                        boundary_analysis['boundary_accuracy'] + 
                        classification_analysis['classification_accuracy']) / 3
        
        print(f"\n🎯 Overall Pipeline Score: {overall_score:.1f}%")
        
        if overall_score >= 90:
            print("🎉 Excellent! Pipeline performance is very high.")
        elif overall_score >= 75:
            print("👍 Good! Pipeline performance is acceptable with room for improvement.")
        elif overall_score >= 50:
            print("⚠️  Fair! Pipeline needs significant improvements.")
        else:
            print("🚨 Poor! Pipeline requires major overhaul.")
            
    except Exception as e:
        print(f"❌ Error during pipeline testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()