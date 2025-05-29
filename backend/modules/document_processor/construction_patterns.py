"""Construction-specific document patterns for boundary detection.

This module contains patterns specifically tuned for construction documents
that may have poor OCR quality.
"""

import re

# Email patterns that work with OCR errors
EMAIL_PATTERNS = [
    # Standard patterns
    re.compile(r"From:\s*.*@", re.IGNORECASE),
    re.compile(r"To:\s*.*@", re.IGNORECASE),
    re.compile(r"Subject:", re.IGNORECASE),
    re.compile(r"Sent:", re.IGNORECASE),
    
    # OCR-friendly patterns (common misreads)
    re.compile(r"(From|Form|Fram):\s*\w+", re.IGNORECASE),
    re.compile(r"(To|Ta|lo):\s*\w+", re.IGNORECASE),
    re.compile(r"(Subject|Subjact|Subjet):", re.IGNORECASE),
    
    # Date patterns for emails
    re.compile(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", re.IGNORECASE),
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}", re.IGNORECASE),
    re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)", re.IGNORECASE),
]

# Submittal/Transmittal patterns
SUBMITTAL_PATTERNS = [
    re.compile(r"SUBMITTAL\s*TRANSMITTAL", re.IGNORECASE),
    re.compile(r"SUBMITTAL", re.IGNORECASE),
    re.compile(r"TRANSMITTAL", re.IGNORECASE),
    re.compile(r"Reference\s*Number", re.IGNORECASE),
    re.compile(r"Transmitted\s*(To|For)", re.IGNORECASE),
    
    # OCR variants
    re.compile(r"SUBM[I1]TTAL", re.IGNORECASE),
    re.compile(r"TRANSM[I1]TTAL", re.IGNORECASE),
]

# Payment document patterns
PAYMENT_PATTERNS = [
    re.compile(r"APPLICATION\s+AND\s+CERTIFICATE\s+FOR\s+PAYMENT", re.IGNORECASE),
    re.compile(r"SCHEDULE\s+OF\s+VALUES", re.IGNORECASE),
    re.compile(r"CONTINUATION\s+SHEET", re.IGNORECASE),
    re.compile(r"AIA\s+Document", re.IGNORECASE),
    re.compile(r"Application\s+No", re.IGNORECASE),
    
    # OCR variants
    re.compile(r"APPL[I1]CAT[I1]ON.*PAYMENT", re.IGNORECASE),
    re.compile(r"CERT[I1]F[I1]CATE.*PAYMENT", re.IGNORECASE),
    re.compile(r"A[I1]A\s+Document", re.IGNORECASE),
]

# Shipping/Packing patterns
SHIPPING_PATTERNS = [
    re.compile(r"PACKING\s+SLIP", re.IGNORECASE),
    re.compile(r"SALES\s+ORDER", re.IGNORECASE),
    re.compile(r"Ship\s+To", re.IGNORECASE),
    re.compile(r"Sold\s+To", re.IGNORECASE),
    re.compile(r"Customer\s+PO", re.IGNORECASE),
    
    # OCR variants
    re.compile(r"PACK[I1]NG\s+SL[I1]P", re.IGNORECASE),
]

# RFI patterns
RFI_PATTERNS = [
    re.compile(r"REQUEST\s+FOR\s+INFORMATION", re.IGNORECASE),
    re.compile(r"RFI\s*#?\s*\d+", re.IGNORECASE),
    re.compile(r"Request\s*#", re.IGNORECASE),
    
    # OCR variants
    re.compile(r"RF[I1]\s*#?\s*\d+", re.IGNORECASE),
    re.compile(r"REQUEST.*[I1]NFORMAT[I1]ON", re.IGNORECASE),
]

# Cost proposal patterns
COST_PATTERNS = [
    re.compile(r"COST\s+PROPOSAL", re.IGNORECASE),
    re.compile(r"PROPOSAL\s*#", re.IGNORECASE),
    re.compile(r"Quote\s+valid", re.IGNORECASE),
    re.compile(r"Total\s+Cost", re.IGNORECASE),
]

# Change order patterns
CHANGE_ORDER_PATTERNS = [
    re.compile(r"CHANGE\s+ORDER", re.IGNORECASE),
    re.compile(r"C\.?O\.?\s*#?\s*\d+", re.IGNORECASE),
    re.compile(r"PCO\s*#?\s*\d+", re.IGNORECASE),
]

# Drawing/Plan patterns
DRAWING_PATTERNS = [
    re.compile(r"^[A-Z]+\d+\.\d+\s*$", re.MULTILINE),  # Drawing numbers
    re.compile(r"SCALE:", re.IGNORECASE),
    re.compile(r"SHEET\s+\d+\s+OF\s+\d+", re.IGNORECASE),
    re.compile(r"DETAIL", re.IGNORECASE),
    re.compile(r"SECTION", re.IGNORECASE),
    re.compile(r"ELEVATION", re.IGNORECASE),
    re.compile(r"PLAN", re.IGNORECASE),
]

def detect_document_type(text: str) -> list:
    """Detect document type indicators in text.
    
    Returns list of detected document types.
    """
    detected = []
    
    # Check each pattern group
    pattern_groups = [
        ("EMAIL", EMAIL_PATTERNS),
        ("SUBMITTAL", SUBMITTAL_PATTERNS),
        ("PAYMENT", PAYMENT_PATTERNS),
        ("SHIPPING", SHIPPING_PATTERNS),
        ("RFI", RFI_PATTERNS),
        ("COST_PROPOSAL", COST_PATTERNS),
        ("CHANGE_ORDER", CHANGE_ORDER_PATTERNS),
        ("DRAWING", DRAWING_PATTERNS),
    ]
    
    for doc_type, patterns in pattern_groups:
        for pattern in patterns:
            if pattern.search(text[:1500]):  # Check first 1500 chars
                detected.append(doc_type)
                break  # Only need one match per type
    
    return detected

def is_strong_document_start(text: str, page_num: int = 0) -> bool:
    """Check if this page is a strong candidate for document start.
    
    This is more lenient than the standard patterns to handle OCR issues.
    """
    # Very short text is not a document start
    if len(text.strip()) < 20:
        return False
    
    # Check for any document type indicators
    doc_types = detect_document_type(text)
    if doc_types:
        return True
    
    # Check for page 1 indicators
    if re.search(r"Page\s*1\s*(of|/)", text, re.IGNORECASE):
        return True
    
    # Check for strong header patterns (all caps titles)
    lines = text.split('\n')
    for i, line in enumerate(lines[:5]):  # Check first 5 lines
        if len(line.strip()) > 10 and line.isupper() and len(line.split()) <= 5:
            # Likely a document title
            return True
    
    return False