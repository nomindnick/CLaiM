#!/usr/bin/env python3
"""
Comprehensive OCR engine testing script.

Tests multiple OCR engines on Test_PDF_Set_1.pdf to find the best approach
for CPRA-style documents.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

import fitz
import numpy as np
from PIL import Image
import cv2

# Import our existing OCR handler
from modules.document_processor.ocr_handler import OCRHandler
from modules.document_processor.improved_ocr_handler import ImprovedOCRHandler

# Ground truth for testing (same pages as boundary analysis)
KEY_PAGES = [1, 5, 7, 9, 13, 14, 18, 20, 23, 26, 32, 34, 35, 36]  # 0-indexed
EXPECTED_DOCS = [
    "Email Chain", "Email Chain", "Submittal", "Schedule of Values", 
    "Email", "Application for Payment", "Invoice", "Invoice", 
    "Request for Information", "Plans and Specifications", 
    "Cost Proposal", "Cost Proposal", "Cost Proposal", "Email"
]


class MultiOCRTester:
    """Test multiple OCR engines and compare results."""
    
    def __init__(self):
        """Initialize all available OCR engines."""
        self.engines = {}
        
        # Always available: Tesseract via our handlers
        self.engines['tesseract_original'] = OCRHandler()
        self.engines['tesseract_improved'] = ImprovedOCRHandler()
        
        # Try to load EasyOCR
        try:
            import easyocr
            self.engines['easyocr'] = self._create_easyocr_handler()
            print("✅ EasyOCR loaded successfully")
        except ImportError:
            print("⚠️  EasyOCR not available")
        
        # Try to load PaddleOCR
        try:
            import paddleocr
            self.engines['paddleocr'] = self._create_paddleocr_handler()
            print("✅ PaddleOCR loaded successfully")
        except ImportError:
            print("⚠️  PaddleOCR not available")
        
        print(f"🔧 Loaded {len(self.engines)} OCR engines: {list(self.engines.keys())}")
    
    def _create_easyocr_handler(self):
        """Create EasyOCR handler."""
        import easyocr
        
        class EasyOCRHandler:
            def __init__(self):
                self.reader = easyocr.Reader(['en'], gpu=False)  # Use CPU to avoid GPU issues
            
            def process_page(self, page: fitz.Page, dpi: int = 300) -> Tuple[str, float]:
                # Convert page to image
                mat = fitz.Matrix(dpi/72.0, dpi/72.0)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
                
                # Convert to numpy array for EasyOCR
                img_array = np.array(img)
                
                # Run EasyOCR
                results = self.reader.readtext(img_array, detail=1)
                
                # Extract text and confidence
                texts = []
                confidences = []
                
                for (bbox, text, conf) in results:
                    if text.strip():
                        texts.append(text.strip())
                        confidences.append(conf)
                
                full_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                return full_text, avg_confidence
        
        return EasyOCRHandler()
    
    def _create_paddleocr_handler(self):
        """Create PaddleOCR handler."""
        from paddleocr import PaddleOCR
        
        class PaddleOCRHandler:
            def __init__(self):
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            
            def process_page(self, page: fitz.Page, dpi: int = 300) -> Tuple[str, float]:
                # Convert page to image
                mat = fitz.Matrix(dpi/72.0, dpi/72.0)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Run PaddleOCR
                results = self.ocr.ocr(img_array, cls=True)
                
                # Extract text and confidence
                texts = []
                confidences = []
                
                if results and results[0]:
                    for line in results[0]:
                        if line and len(line) >= 2:
                            text = line[1][0]  # Text content
                            conf = line[1][1]  # Confidence
                            if text.strip():
                                texts.append(text.strip())
                                confidences.append(conf)
                
                full_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                return full_text, avg_confidence
        
        return PaddleOCRHandler()
    
    def test_single_page(self, pdf_doc: fitz.Document, page_num: int) -> Dict[str, Dict[str, Any]]:
        """Test all OCR engines on a single page."""
        page = pdf_doc[page_num]
        results = {}
        
        print(f"\n📄 Testing Page {page_num + 1}")
        print("-" * 40)
        
        for engine_name, engine in self.engines.items():
            print(f"🔧 Testing {engine_name}...")
            
            try:
                start_time = time.time()
                text, confidence = engine.process_page(page, dpi=300)
                processing_time = time.time() - start_time
                
                results[engine_name] = {
                    'text': text,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'text_length': len(text),
                    'word_count': len(text.split()) if text else 0,
                    'success': True
                }
                
                print(f"   ✅ Success: {len(text)} chars, confidence: {confidence:.3f}, time: {processing_time:.2f}s")
                
                # Show first few words for quick verification
                first_words = ' '.join(text.split()[:10]) if text else "(empty)"
                print(f"   📝 First words: {first_words}...")
                
            except Exception as e:
                results[engine_name] = {
                    'text': '',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'text_length': 0,
                    'word_count': 0,
                    'success': False,
                    'error': str(e)
                }
                print(f"   ❌ Failed: {str(e)}")
        
        return results
    
    def analyze_boundary_patterns(self, text: str, expected_doc_type: str) -> Dict[str, int]:
        """Analyze if text contains expected boundary patterns."""
        text_lower = text.lower()
        patterns_found = {}
        
        # Email patterns
        email_patterns = ['from:', 'to:', 'subject:', '@', 'sent:', 'dear', 'sincerely']
        patterns_found['email'] = sum(1 for p in email_patterns if p in text_lower)
        
        # Document type patterns
        submittal_patterns = ['submittal', 'transmittal', 'attached', 'review']
        patterns_found['submittal'] = sum(1 for p in submittal_patterns if p in text_lower)
        
        schedule_patterns = ['schedule', 'values', 'line item', 'total']
        patterns_found['schedule'] = sum(1 for p in schedule_patterns if p in text_lower)
        
        payment_patterns = ['application', 'payment', 'period ending', 'amount due']
        patterns_found['payment'] = sum(1 for p in payment_patterns if p in text_lower)
        
        invoice_patterns = ['invoice', 'billing', 'packing', 'sales order', 'ship to']
        patterns_found['invoice'] = sum(1 for p in invoice_patterns if p in text_lower)
        
        rfi_patterns = ['request', 'information', 'rfi', 'clarification', 'response required']
        patterns_found['rfi'] = sum(1 for p in rfi_patterns if p in text_lower)
        
        cost_patterns = ['cost', 'proposal', 'estimate', 'quote']
        patterns_found['cost'] = sum(1 for p in cost_patterns if p in text_lower)
        
        # Match against expected type
        type_mapping = {
            'Email Chain': 'email',
            'Email': 'email',
            'Submittal': 'submittal',
            'Schedule of Values': 'schedule',
            'Application for Payment': 'payment',
            'Invoice': 'invoice',
            'Request for Information': 'rfi',
            'Cost Proposal': 'cost'
        }
        
        expected_pattern = type_mapping.get(expected_doc_type, 'unknown')
        patterns_found['expected_match'] = patterns_found.get(expected_pattern, 0)
        
        return patterns_found
    
    def run_comprehensive_test(self, pdf_path: str) -> Dict[str, Any]:
        """Run comprehensive test on all key pages."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Test PDF not found: {pdf_path}")
        
        print(f"🔍 Starting comprehensive OCR engine test")
        print(f"📄 PDF: {pdf_path}")
        print("=" * 80)
        
        pdf_doc = fitz.open(pdf_path)
        total_pages = pdf_doc.page_count
        
        print(f"📊 Testing {len(KEY_PAGES)} key pages out of {total_pages} total pages")
        
        all_results = {}
        engine_summaries = {engine: {'total_confidence': 0, 'total_chars': 0, 'total_time': 0, 'successes': 0, 'pattern_matches': 0} 
                          for engine in self.engines.keys()}
        
        for i, page_num in enumerate(KEY_PAGES):
            if page_num >= total_pages:
                continue
            
            expected_doc = EXPECTED_DOCS[i]
            page_results = self.test_single_page(pdf_doc, page_num)
            
            # Analyze pattern matching for each engine
            for engine_name, result in page_results.items():
                if result['success']:
                    patterns = self.analyze_boundary_patterns(result['text'], expected_doc)
                    result['patterns'] = patterns
                    
                    # Update summaries
                    summary = engine_summaries[engine_name]
                    summary['total_confidence'] += result['confidence']
                    summary['total_chars'] += result['text_length']
                    summary['total_time'] += result['processing_time']
                    summary['successes'] += 1
                    summary['pattern_matches'] += patterns['expected_match']
            
            all_results[f"page_{page_num + 1}"] = {
                'expected_type': expected_doc,
                'results': page_results
            }
        
        pdf_doc.close()
        
        # Calculate final summaries
        for engine_name, summary in engine_summaries.items():
            if summary['successes'] > 0:
                summary['avg_confidence'] = summary['total_confidence'] / summary['successes']
                summary['avg_chars'] = summary['total_chars'] / summary['successes']
                summary['avg_time'] = summary['total_time'] / summary['successes']
                summary['pattern_match_rate'] = summary['pattern_matches'] / len(KEY_PAGES)
            else:
                summary['avg_confidence'] = 0
                summary['avg_chars'] = 0
                summary['avg_time'] = 0
                summary['pattern_match_rate'] = 0
        
        return {
            'page_results': all_results,
            'engine_summaries': engine_summaries,
            'test_metadata': {
                'pdf_path': pdf_path,
                'total_pages': total_pages,
                'tested_pages': len(KEY_PAGES),
                'engines_tested': list(self.engines.keys())
            }
        }
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print a comprehensive summary report."""
        print("\n" + "=" * 80)
        print("📊 OCR ENGINE COMPARISON REPORT")
        print("=" * 80)
        
        summaries = results['engine_summaries']
        
        # Rank engines by overall performance
        engine_scores = {}
        for engine, summary in summaries.items():
            # Weighted score: confidence (40%) + pattern matching (40%) + speed (20%)
            conf_score = summary['avg_confidence'] * 0.4
            pattern_score = summary['pattern_match_rate'] * 0.4
            speed_score = (1.0 / max(summary['avg_time'], 0.1)) * 0.2  # Inverse of time
            
            engine_scores[engine] = conf_score + pattern_score + speed_score
        
        ranked_engines = sorted(engine_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 RANKING (Best to Worst):")
        print("-" * 50)
        
        for rank, (engine, score) in enumerate(ranked_engines, 1):
            summary = summaries[engine]
            print(f"{rank}. {engine.upper()}")
            print(f"   Overall Score: {score:.3f}")
            print(f"   Avg Confidence: {summary['avg_confidence']:.3f}")
            print(f"   Pattern Match Rate: {summary['pattern_match_rate']:.2f}")
            print(f"   Avg Processing Time: {summary['avg_time']:.2f}s")
            print(f"   Avg Text Length: {summary['avg_chars']:.0f} chars")
            print(f"   Success Rate: {summary['successes']}/{len(KEY_PAGES)}")
            print()
        
        # Best engine recommendation
        best_engine = ranked_engines[0][0]
        print(f"🎯 RECOMMENDATION: Use {best_engine.upper()} for production")
        
        best_summary = summaries[best_engine]
        print(f"   Expected confidence: {best_summary['avg_confidence']:.1%}")
        print(f"   Expected pattern detection: {best_summary['pattern_match_rate']:.1%}")
        print(f"   Expected processing time: {best_summary['avg_time']:.1f}s per page")
        
        return best_engine


def main():
    """Main test execution."""
    test_pdf_path = "/home/nick/Projects/CLaiM/tests/Test_PDF_Set_1.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"❌ Error: Test PDF not found at {test_pdf_path}")
        return
    
    # Initialize tester
    print("🚀 Initializing OCR engine tester...")
    tester = MultiOCRTester()
    
    if not tester.engines:
        print("❌ No OCR engines available!")
        return
    
    try:
        # Run comprehensive test
        results = tester.run_comprehensive_test(test_pdf_path)
        
        # Print summary report
        best_engine = tester.print_summary_report(results)
        
        # Additional analysis
        print(f"\n🔍 DETAILED ANALYSIS")
        print("-" * 50)
        
        # Find most problematic pages
        problem_pages = []
        for page_key, page_data in results['page_results'].items():
            page_num = page_key.split('_')[1]
            max_confidence = max(
                (r['confidence'] for r in page_data['results'].values() if r['success']),
                default=0
            )
            if max_confidence < 0.7:  # Low confidence threshold
                problem_pages.append((page_num, max_confidence, page_data['expected_type']))
        
        if problem_pages:
            print(f"\n⚠️  PROBLEMATIC PAGES (confidence < 70%):")
            for page_num, conf, doc_type in problem_pages:
                print(f"   Page {page_num}: {conf:.1%} confidence ({doc_type})")
        else:
            print(f"\n✅ All pages had reasonable confidence levels!")
        
        # Save detailed results
        import json
        output_file = "/home/nick/Projects/CLaiM/ocr_engine_comparison_results.json"
        with open(output_file, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {
                'summary': summaries,
                'best_engine': best_engine,
                'test_metadata': results['test_metadata']
            }
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()