#!/usr/bin/env python3
"""Debug boundary detection to see what boundaries are found vs missing pages."""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

import fitz  # PyMuPDF
from modules.document_processor.hybrid_boundary_detector import HybridBoundaryDetector, DetectionLevel

def main():
    print("=== Debugging Boundary Detection ===\n")
    
    test_pdf = Path("tests/Test_PDF_Set_2.pdf")
    if not test_pdf.exists():
        print("❌ Test PDF not found")
        return
    
    # Open PDF
    pdf_doc = fitz.open(str(test_pdf))
    total_pages = pdf_doc.page_count
    print(f"PDF has {total_pages} pages")
    
    # Test boundary detection at different levels
    detector = HybridBoundaryDetector()
    
    for level in [DetectionLevel.HEURISTIC, DetectionLevel.VISUAL, DetectionLevel.DEEP]:
        print(f"\n=== Testing {level.name} Detection ===")
        
        try:
            result = detector.detect_boundaries(pdf_doc, level)
            boundaries = result.boundaries
            
            print(f"Detection level: {result.detection_level.name}")
            print(f"Found {len(boundaries)} boundaries")
            print(f"Confidence scores: {result.confidence_scores}")
            
            if boundaries:
                print("Boundaries found:")
                total_covered = 0
                all_covered_pages = set()
                
                for i, (start, end) in enumerate(boundaries):
                    page_count = end - start + 1
                    total_covered += page_count
                    pages = list(range(start + 1, end + 2))  # Convert to 1-indexed
                    all_covered_pages.update(pages)
                    print(f"  Document {i+1}: pages {start+1}-{end+1} ({page_count} pages)")
                
                print(f"\nCoverage summary:")
                print(f"  Total pages covered: {total_covered}/{total_pages}")
                print(f"  Coverage percentage: {total_covered/total_pages*100:.1f}%")
                
                # Find missing pages
                all_pages = set(range(1, total_pages + 1))
                missing_pages = all_pages - all_covered_pages
                if missing_pages:
                    print(f"  Missing pages: {sorted(missing_pages)}")
                    
                    # Analyze gaps
                    print(f"\nGap analysis:")
                    sorted_boundaries = sorted(boundaries)
                    
                    # Check gap before first document
                    if sorted_boundaries and sorted_boundaries[0][0] > 0:
                        gap_pages = list(range(1, sorted_boundaries[0][0] + 1))
                        print(f"  Gap before first doc: pages {gap_pages}")
                    
                    # Check gaps between documents
                    for i in range(len(sorted_boundaries) - 1):
                        current_end = sorted_boundaries[i][1]
                        next_start = sorted_boundaries[i + 1][0]
                        
                        if next_start > current_end + 1:
                            gap_pages = list(range(current_end + 2, next_start + 1))
                            print(f"  Gap between docs {i+1} and {i+2}: pages {gap_pages}")
                    
                    # Check gap after last document
                    if sorted_boundaries and sorted_boundaries[-1][1] < total_pages - 1:
                        gap_pages = list(range(sorted_boundaries[-1][1] + 2, total_pages + 1))
                        print(f"  Gap after last doc: pages {gap_pages}")
                else:
                    print("  ✅ All pages covered!")
            
            else:
                print("  No boundaries found")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    pdf_doc.close()
    
    print(f"\n=== Recommendations ===")
    print("If many pages are missing, consider:")
    print("1. Using a more permissive boundary detection threshold")
    print("2. Implementing a 'gap filling' algorithm that treats uncovered pages as separate documents")
    print("3. Adding fallback logic to merge adjacent pages that don't match clear document patterns")

if __name__ == "__main__":
    main()