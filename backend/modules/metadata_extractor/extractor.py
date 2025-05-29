"""Main metadata extraction class that coordinates pattern matching."""

import re
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import defaultdict

from loguru import logger

from modules.document_processor.models import Document, DocumentMetadata, Party
from .patterns import PatternMatcher
from .normalizer import EntityNormalizer


class MetadataExtractor:
    """Extracts structured metadata from construction documents."""
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.entity_normalizer = EntityNormalizer()
    
    def extract_metadata(self, document: Document) -> DocumentMetadata:
        """Extract all metadata from a document.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Extracted metadata
        """
        logger.info(f"Extracting metadata from document {document.id}")
        
        # Use full document text for extraction
        text = document.text
        
        # Extract different types of metadata
        dates = self._extract_and_filter_dates(text)
        reference_numbers = self.pattern_matcher.extract_reference_numbers(text)
        amounts = self.pattern_matcher.extract_amounts(text)
        parties = self._extract_and_normalize_parties(text)
        
        # Try to determine document date
        document_date = self._determine_document_date(text, dates)
        
        # Extract project information
        project_info = self._extract_project_info(text, reference_numbers)
        
        # Determine subject/title
        subject = self._extract_subject(document, reference_numbers)
        
        # Create metadata object
        metadata = DocumentMetadata(
            title=subject,
            date=document_date,
            parties=parties,
            reference_numbers=self._flatten_reference_numbers(reference_numbers),
            amounts=amounts,
            keywords=self._extract_keywords(text),
            project_number=project_info.get("number"),
            project_name=project_info.get("name"),
            subject=subject
        )
        
        # Log extraction results
        logger.info(f"Extracted metadata - Dates: {len(dates)}, Parties: {len(parties)}, "
                   f"References: {len(reference_numbers)}, Amounts: {len(amounts)}")
        
        return metadata
    
    def _extract_and_filter_dates(self, text: str) -> List[datetime]:
        """Extract dates and filter out unlikely ones."""
        dates = self.pattern_matcher.extract_dates(text)
        
        # Filter dates
        filtered = []
        current_year = datetime.now().year
        
        for date in dates:
            # Skip dates too far in the future or past
            if 1990 <= date.year <= current_year + 5:
                filtered.append(date)
        
        return filtered
    
    def _extract_and_normalize_parties(self, text: str) -> List[Party]:
        """Extract and normalize party information."""
        raw_parties = self.pattern_matcher.extract_parties(text)
        
        # Group by normalized name
        party_map = defaultdict(lambda: {"roles": set(), "emails": set(), "phones": set()})
        
        for party_info in raw_parties:
            name = party_info.get("name", "")
            if not name:
                continue
            
            # Normalize the name
            normalized_name = self.entity_normalizer.normalize_party_name(name)
            
            # Collect information
            if "role" in party_info:
                party_map[normalized_name]["roles"].add(party_info["role"])
            if "email" in party_info:
                party_map[normalized_name]["emails"].add(party_info["email"])
            if "phone" in party_info:
                party_map[normalized_name]["phones"].add(party_info["phone"])
            if "type" in party_info:
                party_map[normalized_name]["type"] = party_info["type"]
        
        # Convert to Party objects
        parties = []
        for name, info in party_map.items():
            party = Party(
                name=name,
                role=", ".join(info["roles"]) if info["roles"] else None,
                email=list(info["emails"])[0] if info["emails"] else None,
                phone=list(info["phones"])[0] if info["phones"] else None,
            )
            
            # Set company if it's a company
            if info.get("type") == "company":
                party.company = name
            
            parties.append(party)
        
        return parties
    
    def _determine_document_date(self, text: str, dates: List[datetime]) -> Optional[datetime]:
        """Try to determine the main document date."""
        if not dates:
            return None
        
        # Look for dates near common date indicators
        date_indicators = [
            "date:", "dated:", "date of", "as of", "effective date",
            "issued:", "prepared:", "submitted:"
        ]
        
        text_lower = text.lower()
        
        for date in dates:
            date_str = date.strftime("%m/%d/%Y")
            date_pos = text.find(date_str)
            
            if date_pos == -1:
                # Try other formats
                date_str = date.strftime("%B %d, %Y")
                date_pos = text.find(date_str)
            
            if date_pos != -1:
                # Check if near a date indicator
                context_start = max(0, date_pos - 50)
                context = text_lower[context_start:date_pos]
                
                for indicator in date_indicators:
                    if indicator in context:
                        return date
        
        # If no specific document date found, return the most recent date
        return max(dates)
    
    def _extract_project_info(self, text: str, reference_numbers: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract project number and name."""
        info = {}
        
        # Get project number from references
        if "Project" in reference_numbers:
            info["number"] = reference_numbers["Project"][0]
        
        # Try to extract project name
        project_patterns = [
            re.compile(r'Project\s*(?:Name)?:\s*([^\n]+)', re.IGNORECASE),
            re.compile(r'RE:\s*([^\n]+?)(?:\s*-\s*[A-Z]{2,})?$', re.MULTILINE | re.IGNORECASE),
        ]
        
        for pattern in project_patterns:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip()
                # Clean up the name
                name = name.rstrip('.,;')
                if len(name) > 5 and len(name) < 100:  # Reasonable length
                    info["name"] = name
                    break
        
        return info
    
    def _extract_subject(self, document: Document, reference_numbers: Dict[str, List[str]]) -> str:
        """Extract or generate document subject/title."""
        # For RFIs, use the RFI number
        if document.type.value == "rfi" and "RFI" in reference_numbers:
            return f"RFI #{reference_numbers['RFI'][0]}"
        
        # For Change Orders
        if document.type.value == "change_order" and "Change Order" in reference_numbers:
            return f"Change Order #{reference_numbers['Change Order'][0]}"
        
        # For Invoices
        if document.type.value == "invoice" and "Invoice" in reference_numbers:
            return f"Invoice #{reference_numbers['Invoice'][0]}"
        
        # Try to extract from subject line or RE: line
        subject_patterns = [
            re.compile(r'Subject:\s*([^\n]+)', re.IGNORECASE),
            re.compile(r'RE:\s*([^\n]+)', re.IGNORECASE),
        ]
        
        for pattern in subject_patterns:
            match = pattern.search(document.text)
            if match:
                subject = match.group(1).strip()
                # Clean up
                subject = subject.rstrip('.,;')
                if len(subject) > 5 and len(subject) < 200:
                    return subject
        
        # Default to document type
        return f"{document.type.value.replace('_', ' ').title()}"
    
    def _flatten_reference_numbers(self, ref_dict: Dict[str, List[str]]) -> List[str]:
        """Flatten reference numbers dictionary to a list."""
        flattened = []
        for ref_type, numbers in ref_dict.items():
            for number in numbers:
                if ref_type in ["RFI", "ASI", "PCO", "RFP"]:
                    flattened.append(f"{ref_type} {number}")
                else:
                    flattened.append(f"{ref_type} #{number}")
        return flattened
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from document."""
        # Construction-specific keywords to look for
        keyword_patterns = [
            "delay", "impact", "change", "extra work", "differing conditions",
            "acceleration", "disruption", "inefficiency", "overtime", "premium time",
            "notice", "claim", "backcharge", "liquidated damages", "substantial completion",
            "final completion", "punch list", "warranty", "defect", "nonconforming",
            "rejection", "approval", "submittal", "RFI", "change order", "time extension",
            "force majeure", "weather", "unforeseen", "concealed conditions",
            "design error", "coordination", "sequence", "milestone", "critical path",
            "float", "schedule", "baseline", "as-built", "shop drawings", "samples",
            "product data", "safety", "OSHA", "violation", "stop work", "suspension"
        ]
        
        # Find which keywords appear in the text
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in keyword_patterns:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords