#!/usr/bin/env python3
"""Test metadata extraction functionality."""

import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from modules.document_processor.models import Document, DocumentPage, DocumentType, DocumentMetadata
from modules.metadata_extractor import MetadataExtractor
from modules.metadata_extractor.normalizer import EntityNormalizer
from loguru import logger


def test_rfi_extraction():
    """Test metadata extraction from RFI document."""
    print("\n" + "="*60)
    print("TESTING RFI METADATA EXTRACTION")
    print("="*60)
    
    # Sample RFI text
    rfi_text = """
    REQUEST FOR INFORMATION
    
    Date: May 15, 2024
    RFI #: 123
    Project: Lincoln High School Gymnasium Renovation
    Project No: 2024-LHS-001
    
    To: ABC Construction Company, Inc.
    Attention: John Smith, Project Manager
    Email: jsmith@abcconstruction.com
    Phone: (555) 123-4567
    
    From: Los Angeles Unified School District
    Facilities Department
    Contact: Jane Doe
    Email: jdoe@lausd.edu
    
    Subject: Concrete Specifications for Foundation
    
    We are requesting clarification on the concrete specifications for the gymnasium
    foundation. The plans show 3000 PSI concrete, but the specifications indicate
    4000 PSI. Please confirm which is correct.
    
    This RFI requires a response by May 22, 2024 to avoid delays.
    
    Potential cost impact: $15,000.00 to $20,000.00
    
    Also referencing Change Order #005 and ASI #002.
    """
    
    # Create document
    doc = Document(
        id="test-rfi-001",
        source_pdf_id="pdf-001",
        source_pdf_path=Path("/test/rfi.pdf"),
        type=DocumentType.RFI,
        pages=[DocumentPage(page_number=1, text=rfi_text)],
        page_range=(1, 1),
        text=rfi_text,
        metadata=DocumentMetadata()
    )
    
    # Extract metadata
    extractor = MetadataExtractor()
    metadata = extractor.extract_metadata(doc)
    
    # Display results
    print(f"\n📄 Document Title: {metadata.title}")
    print(f"📅 Document Date: {metadata.date}")
    print(f"📋 Subject: {metadata.subject}")
    
    print(f"\n🏗️ Project Info:")
    print(f"   - Number: {metadata.project_number}")
    print(f"   - Name: {metadata.project_name}")
    
    print(f"\n🔢 Reference Numbers:")
    for ref in metadata.reference_numbers:
        print(f"   - {ref}")
    
    print(f"\n👥 Parties:")
    for party in metadata.parties:
        print(f"   - {party.name}")
        if party.role:
            print(f"     Role: {party.role}")
        if party.email:
            print(f"     Email: {party.email}")
        if party.phone:
            print(f"     Phone: {party.phone}")
    
    print(f"\n💰 Amounts:")
    for amount in metadata.amounts:
        print(f"   - ${amount:,.2f}")
    
    print(f"\n🏷️ Keywords:")
    for keyword in metadata.keywords[:10]:  # Show first 10
        print(f"   - {keyword}")


def test_change_order_extraction():
    """Test metadata extraction from change order."""
    print("\n" + "="*60)
    print("TESTING CHANGE ORDER METADATA EXTRACTION")
    print("="*60)
    
    co_text = """
    CHANGE ORDER
    
    Change Order No: 007
    Date: June 1, 2024
    Contract #: C-2024-100
    
    Owner: Orange County School District
    700 Kalmus Drive
    Costa Mesa, CA 92626
    
    Contractor: Metro Builders Inc.
    Attn: Mike Johnson, Project Manager
    mjohnson@metrobuilders.com
    (714) 555-1234
    
    Architect: Smith & Associates Architecture LLC
    
    Project: Riverside Elementary School Addition
    Project Number: 2024-RES-001
    
    Description of Change:
    Additional structural reinforcement required due to unforeseen soil conditions
    discovered during excavation. This is a differing site condition per contract
    section 4.3.
    
    Schedule Impact: 15 calendar days time extension requested
    
    Cost Breakdown:
    Labor: $45,000.00
    Materials: $25,890.50
    Equipment: $5,000.00
    
    Original Contract Amount: $1,234,567.00
    Previous Change Orders (1-6): $50,000.00
    This Change Order: $75,890.50
    New Contract Total: $1,360,457.50
    
    References:
    - RFI #089 dated 5/15/2024
    - Geotechnical Report Amendment dated 5/20/2024
    """
    
    # Create document
    doc = Document(
        id="test-co-001",
        source_pdf_id="pdf-002",
        source_pdf_path=Path("/test/co.pdf"),
        type=DocumentType.CHANGE_ORDER,
        pages=[DocumentPage(page_number=1, text=co_text)],
        page_range=(1, 1),
        text=co_text
    )
    
    # Extract metadata
    extractor = MetadataExtractor()
    metadata = extractor.extract_metadata(doc)
    
    # Display results
    print(f"\n📄 Document Title: {metadata.title}")
    print(f"📅 Document Date: {metadata.date}")
    
    print(f"\n🔢 Reference Numbers:")
    for ref in metadata.reference_numbers:
        print(f"   - {ref}")
    
    print(f"\n👥 Parties:")
    for party in metadata.parties:
        print(f"   - {party.name}")
        if party.email:
            print(f"     Email: {party.email}")
    
    print(f"\n💰 Amounts:")
    for amount in sorted(metadata.amounts, reverse=True):
        print(f"   - ${amount:,.2f}")
    
    print(f"\n🏷️ Keywords Found:")
    for keyword in metadata.keywords:
        print(f"   - {keyword}")


def test_party_normalization():
    """Test party name normalization."""
    print("\n" + "="*60)
    print("TESTING PARTY NAME NORMALIZATION")
    print("="*60)
    
    normalizer = EntityNormalizer()
    
    # Test cases
    test_names = [
        "ABC Construction Inc.",
        "ABC Construction Incorporated",
        "ABC Const. Inc",
        "abc construction inc",
        "Smith & Associates LLC",
        "Smith and Associates L.L.C.",
        "Metro Engineering Corp.",
        "Metro Eng. Corporation",
        "Los Angeles Unified School District",
        "LAUSD"
    ]
    
    print("\nNormalization Results:")
    for name in test_names:
        normalized = normalizer.normalize_party_name(name)
        print(f"   '{name}' → '{normalized}'")
    
    # Test similarity detection
    print("\nSimilarity Groups:")
    similar_groups = normalizer.find_similar_parties(test_names)
    for canonical, variations in similar_groups.items():
        print(f"\n   Canonical: {canonical}")
        for var in variations:
            if var != canonical:
                print(f"      - {var}")


def test_pattern_matching():
    """Test individual pattern matching."""
    print("\n" + "="*60)
    print("TESTING PATTERN MATCHING")
    print("="*60)
    
    from modules.metadata_extractor.patterns import PatternMatcher
    matcher = PatternMatcher()
    
    # Test text with various patterns
    test_text = """
    Invoice #2024-0005
    Date: December 15, 2024
    Due Date: Jan 15, 2025
    
    Bill To:
    XYZ School District
    123 Education Blvd
    Los Angeles, CA 90001
    
    For: Construction Services per Contract C-2024-100
    Project: Lincoln High School Renovation (Project #2024-LHS-001)
    
    Previous Balance: $500,000.00
    This Invoice: $125,456.78
    Total Due: $625,456.78
    
    Contact: billing@abcconstruction.com or call (555) 987-6543
    """
    
    print("\nDates Found:")
    dates = matcher.extract_dates(test_text)
    for date in dates:
        print(f"   - {date.strftime('%B %d, %Y')}")
    
    print("\nReference Numbers:")
    refs = matcher.extract_reference_numbers(test_text)
    for ref_type, numbers in refs.items():
        print(f"   {ref_type}: {', '.join(numbers)}")
    
    print("\nAmounts:")
    amounts = matcher.extract_amounts(test_text)
    for amount in amounts:
        print(f"   - ${amount:,.2f}")
    
    print("\nParties/Entities:")
    parties = matcher.extract_parties(test_text)
    for party in parties:
        print(f"   - {party}")


def main():
    """Run all tests."""
    print("\n🧪 METADATA EXTRACTION TEST SUITE")
    print("=" * 60)
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run tests
    test_rfi_extraction()
    test_change_order_extraction()
    test_party_normalization()
    test_pattern_matching()
    
    print("\n✅ All tests completed!")
    print("\nNext steps:")
    print("1. Run pytest for unit tests: pytest backend/modules/metadata_extractor/tests/")
    print("2. Test with real PDFs using the API endpoint")
    print("3. Integrate with the document processing pipeline")


if __name__ == "__main__":
    main()