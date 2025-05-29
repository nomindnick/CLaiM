"""Pattern definitions and matching for construction document metadata."""

import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from loguru import logger


@dataclass
class ExtractionPattern:
    """Pattern definition for metadata extraction."""
    name: str
    pattern: re.Pattern
    group_names: List[str]
    post_processor: Optional[callable] = None


class PatternMatcher:
    """Matches patterns in construction documents to extract metadata."""
    
    def __init__(self):
        self.date_patterns = self._compile_date_patterns()
        self.reference_patterns = self._compile_reference_patterns()
        self.amount_patterns = self._compile_amount_patterns()
        self.party_patterns = self._compile_party_patterns()
        self.entity_indicators = self._load_entity_indicators()
    
    def _compile_date_patterns(self) -> List[ExtractionPattern]:
        """Compile date extraction patterns."""
        patterns = [
            # MM/DD/YYYY or MM-DD-YYYY
            ExtractionPattern(
                name="us_date",
                pattern=re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'),
                group_names=["month", "day", "year"],
                post_processor=self._parse_us_date
            ),
            # YYYY-MM-DD (ISO format)
            ExtractionPattern(
                name="iso_date",
                pattern=re.compile(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b'),
                group_names=["year", "month", "day"],
                post_processor=self._parse_iso_date
            ),
            # Month DD, YYYY or Month DDth, YYYY
            ExtractionPattern(
                name="text_date",
                pattern=re.compile(
                    r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
                    r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+'
                    r'(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
                    re.IGNORECASE
                ),
                group_names=["month", "day", "year"],
                post_processor=self._parse_text_date
            ),
            # DD Month YYYY
            ExtractionPattern(
                name="text_date_euro",
                pattern=re.compile(
                    r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|'
                    r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{4})\b',
                    re.IGNORECASE
                ),
                group_names=["day", "month", "year"],
                post_processor=self._parse_text_date_euro
            )
        ]
        return patterns
    
    def _compile_reference_patterns(self) -> Dict[str, ExtractionPattern]:
        """Compile reference number patterns."""
        patterns = {
            "RFI": ExtractionPattern(
                name="rfi",
                pattern=re.compile(r'\bRFI\s*#?\s*(\d+)\b', re.IGNORECASE),
                group_names=["number"]
            ),
            "Change Order": ExtractionPattern(
                name="change_order",
                pattern=re.compile(r'\b(?:Change\s+Order|CO)\s*#?\s*(\d+)\b', re.IGNORECASE),
                group_names=["number"]
            ),
            "Invoice": ExtractionPattern(
                name="invoice",
                pattern=re.compile(r'\bInvoice\s*#?\s*(\d+)\b', re.IGNORECASE),
                group_names=["number"]
            ),
            "Project": ExtractionPattern(
                name="project",
                pattern=re.compile(r'\bProject\s*(?:#|No\.?|Number):\s*([A-Z0-9-]+)\b', re.IGNORECASE),
                group_names=["number"]
            ),
            "Contract": ExtractionPattern(
                name="contract",
                pattern=re.compile(r'\bContract\s*(?:#|No\.?|Number)?\s*([A-Z0-9-]+)\b', re.IGNORECASE),
                group_names=["number"]
            ),
            "Submittal": ExtractionPattern(
                name="submittal",
                pattern=re.compile(r'\bSubmittal\s*#?\s*(\d+)\b', re.IGNORECASE),
                group_names=["number"]
            ),
            "Payment Application": ExtractionPattern(
                name="payment_app",
                pattern=re.compile(r'\b(?:Payment\s+Application|Pay\s+App)\s*#?\s*(\d+)\b', re.IGNORECASE),
                group_names=["number"]
            ),
            "ASI": ExtractionPattern(
                name="asi",
                pattern=re.compile(r'\bASI\s*#?\s*(\d+)\b', re.IGNORECASE),
                group_names=["number"]
            ),
            "RFP": ExtractionPattern(
                name="rfp",
                pattern=re.compile(r'\bRFP\s*#?\s*(\d+)\b', re.IGNORECASE),
                group_names=["number"]
            ),
            "PCO": ExtractionPattern(
                name="pco",
                pattern=re.compile(r'\bPCO\s*#?\s*(\d+)\b', re.IGNORECASE),
                group_names=["number"]
            )
        }
        return patterns
    
    def _compile_amount_patterns(self) -> List[ExtractionPattern]:
        """Compile monetary amount patterns."""
        patterns = [
            # $1,234.56 or $1234.56
            ExtractionPattern(
                name="dollar_amount",
                pattern=re.compile(r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b'),
                group_names=["amount"],
                post_processor=self._parse_dollar_amount
            ),
            # USD 1,234.56
            ExtractionPattern(
                name="usd_amount",
                pattern=re.compile(r'\bUSD\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b', re.IGNORECASE),
                group_names=["amount"],
                post_processor=self._parse_dollar_amount
            ),
            # Written amounts: "One Hundred Thousand Dollars"
            ExtractionPattern(
                name="written_amount",
                pattern=re.compile(
                    r'\b((?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|'
                    r'Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|'
                    r'Eighteen|Nineteen|Twenty|Thirty|Forty|Fifty|Sixty|Seventy|'
                    r'Eighty|Ninety|Hundred|Thousand|Million|Billion|and|'
                    r'\s)+)\s+Dollars?\b',
                    re.IGNORECASE
                ),
                group_names=["amount"],
                post_processor=self._parse_written_amount
            )
        ]
        return patterns
    
    def _compile_party_patterns(self) -> List[ExtractionPattern]:
        """Compile party/entity extraction patterns."""
        patterns = [
            # Email addresses (good indicator of parties)
            ExtractionPattern(
                name="email",
                pattern=re.compile(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'),
                group_names=["email"]
            ),
            # Phone numbers
            ExtractionPattern(
                name="phone",
                pattern=re.compile(r'\b(?:\+?1[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b'),
                group_names=["area", "prefix", "number"]
            ),
            # Construction-specific role patterns
            ExtractionPattern(
                name="contractor",
                pattern=re.compile(r'\b(?:General\s+)?Contractor:\s*([A-Z][A-Za-z\s&.,]+?)(?:\n|$)', re.MULTILINE),
                group_names=["name"]
            ),
            ExtractionPattern(
                name="owner",
                pattern=re.compile(r'\bOwner:\s*([A-Z][A-Za-z\s&.,]+?)(?:\n|$)', re.MULTILINE),
                group_names=["name"]
            ),
            ExtractionPattern(
                name="architect",
                pattern=re.compile(r'\bArchitect:\s*([A-Z][A-Za-z\s&.,]+?)(?:\n|$)', re.MULTILINE),
                group_names=["name"]
            ),
            ExtractionPattern(
                name="engineer",
                pattern=re.compile(r'\bEngineer:\s*([A-Z][A-Za-z\s&.,]+?)(?:\n|$)', re.MULTILINE),
                group_names=["name"]
            ),
            ExtractionPattern(
                name="subcontractor",
                pattern=re.compile(r'\bSubcontractor:\s*([A-Z][A-Za-z\s&.,]+?)(?:\n|$)', re.MULTILINE),
                group_names=["name"]
            )
        ]
        return patterns
    
    def _load_entity_indicators(self) -> Dict[str, Set[str]]:
        """Load indicators for entity recognition."""
        return {
            "company_suffixes": {
                "Inc", "Inc.", "Incorporated", "Corp", "Corp.", "Corporation",
                "LLC", "L.L.C.", "Company", "Co.", "Ltd", "Ltd.", "Limited",
                "LLP", "L.L.P.", "Partnership", "Associates", "Group",
                "Construction", "Builders", "Contractors", "Engineering",
                "Architects", "Design", "Consultants", "Services"
            },
            "role_indicators": {
                "Contractor", "Subcontractor", "Owner", "Architect", "Engineer",
                "Consultant", "Supplier", "Vendor", "Inspector", "Manager",
                "Superintendent", "Developer", "Client"
            },
            "agency_indicators": {
                "District", "School District", "USD", "Department", "City of",
                "County of", "State of", "Division", "Authority", "Board",
                "Commission", "Agency"
            }
        }
    
    def extract_dates(self, text: str) -> List[datetime]:
        """Extract all dates from text."""
        dates = []
        seen_dates = set()
        
        for pattern in self.date_patterns:
            for match in pattern.pattern.finditer(text):
                try:
                    if pattern.post_processor:
                        date = pattern.post_processor(match)
                        if date and date not in seen_dates:
                            dates.append(date)
                            seen_dates.add(date)
                except Exception as e:
                    logger.debug(f"Failed to parse date from {match.group()}: {e}")
        
        return sorted(dates)
    
    def extract_reference_numbers(self, text: str) -> Dict[str, List[str]]:
        """Extract reference numbers by type."""
        references = {}
        
        for ref_type, pattern in self.reference_patterns.items():
            matches = []
            for match in pattern.pattern.finditer(text):
                number = match.group(1)
                if number not in matches:
                    matches.append(number)
            
            if matches:
                references[ref_type] = matches
        
        return references
    
    def extract_amounts(self, text: str) -> List[float]:
        """Extract monetary amounts from text."""
        amounts = []
        seen_amounts = set()
        
        for pattern in self.amount_patterns:
            for match in pattern.pattern.finditer(text):
                try:
                    if pattern.post_processor:
                        amount = pattern.post_processor(match)
                        if amount and amount not in seen_amounts:
                            amounts.append(amount)
                            seen_amounts.add(amount)
                except Exception as e:
                    logger.debug(f"Failed to parse amount from {match.group()}: {e}")
        
        return sorted(amounts, reverse=True)
    
    def extract_parties(self, text: str) -> List[Dict[str, str]]:
        """Extract parties/entities from text."""
        parties = []
        seen_names = set()
        
        # Extract from explicit patterns
        for pattern in self.party_patterns:
            for match in pattern.pattern.finditer(text):
                if pattern.name == "email":
                    email = match.group(1)
                    # Try to extract name from email context
                    name = self._extract_name_near_email(text, match.start())
                    if name and name not in seen_names:
                        parties.append({"name": name, "email": email})
                        seen_names.add(name)
                elif pattern.name in ["contractor", "owner", "architect", "engineer", "subcontractor"]:
                    name = match.group(1).strip()
                    name = self._clean_party_name(name)
                    if name and name not in seen_names:
                        parties.append({"name": name, "role": pattern.name.title()})
                        seen_names.add(name)
        
        # Extract potential company names
        company_names = self._extract_company_names(text)
        for name in company_names:
            if name not in seen_names:
                parties.append({"name": name, "type": "company"})
                seen_names.add(name)
        
        # Extract government agencies
        agency_names = self._extract_agency_names(text)
        for name in agency_names:
            if name not in seen_names:
                parties.append({"name": name, "type": "agency"})
                seen_names.add(name)
        
        return parties
    
    def _parse_us_date(self, match: re.Match) -> Optional[datetime]:
        """Parse US format date (MM/DD/YYYY)."""
        try:
            month = int(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))
            return datetime(year, month, day)
        except ValueError:
            return None
    
    def _parse_iso_date(self, match: re.Match) -> Optional[datetime]:
        """Parse ISO format date (YYYY-MM-DD)."""
        try:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            return datetime(year, month, day)
        except ValueError:
            return None
    
    def _parse_text_date(self, match: re.Match) -> Optional[datetime]:
        """Parse text format date (Month DD, YYYY)."""
        try:
            month_str = match.group(1)
            day = int(match.group(2))
            year = int(match.group(3))
            
            # Convert month name to number
            month_map = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
                'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
                'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
                'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
                'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
                'december': 12, 'dec': 12
            }
            month = month_map.get(month_str.lower())
            
            if month:
                return datetime(year, month, day)
        except (ValueError, KeyError):
            pass
        return None
    
    def _parse_text_date_euro(self, match: re.Match) -> Optional[datetime]:
        """Parse European text format date (DD Month YYYY)."""
        try:
            day = int(match.group(1))
            month_str = match.group(2)
            year = int(match.group(3))
            
            # Convert month name to number
            month_map = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
                'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
                'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
                'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
                'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
                'december': 12, 'dec': 12
            }
            month = month_map.get(month_str.lower())
            
            if month:
                return datetime(year, month, day)
        except (ValueError, KeyError):
            pass
        return None
    
    def _parse_dollar_amount(self, match: re.Match) -> Optional[float]:
        """Parse dollar amount from match."""
        try:
            amount_str = match.group(1).replace(',', '')
            return float(amount_str)
        except ValueError:
            return None
    
    def _parse_written_amount(self, match: re.Match) -> Optional[float]:
        """Parse written amount to numeric value."""
        # This is a simplified implementation
        # In production, you'd want a more comprehensive number parser
        text = match.group(1).lower()
        
        # Simple mapping for common amounts
        if "million" in text:
            if "one" in text:
                return 1000000.0
            elif "two" in text:
                return 2000000.0
            # Add more as needed
        elif "thousand" in text:
            if "hundred" in text:
                return 100000.0
            elif "fifty" in text:
                return 50000.0
            # Add more as needed
        
        return None
    
    def _extract_name_near_email(self, text: str, email_pos: int, window: int = 100) -> Optional[str]:
        """Extract person/company name near an email address."""
        # Look for name before email
        start = max(0, email_pos - window)
        end = email_pos
        context = text[start:end]
        
        # Common patterns: "John Smith <email>" or "From: John Smith"
        patterns = [
            re.compile(r'([A-Z][a-z]+ [A-Z][a-z]+)\s*<?\s*$'),
            re.compile(r'(?:From|To|CC|Cc):\s*([A-Z][a-z]+ [A-Z][a-z]+)\s*$'),
            re.compile(r'([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)\s*$'),  # John D. Smith
        ]
        
        for pattern in patterns:
            match = pattern.search(context)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_company_names(self, text: str) -> List[str]:
        """Extract potential company names from text."""
        companies = []
        
        # Pattern for company names (Capitalized Words + Company Suffix)
        pattern = re.compile(
            r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(' +
            '|'.join(re.escape(suffix) for suffix in self.entity_indicators["company_suffixes"]) +
            r')\b'
        )
        
        for match in pattern.finditer(text):
            name = f"{match.group(1)} {match.group(2)}"
            companies.append(name)
        
        return companies
    
    def _extract_agency_names(self, text: str) -> List[str]:
        """Extract government agency names from text."""
        agencies = []
        
        # Pattern for agency names
        for indicator in self.entity_indicators["agency_indicators"]:
            pattern = re.compile(
                rf'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+{re.escape(indicator)})\b'
            )
            for match in pattern.finditer(text):
                agencies.append(match.group(1))
        
        return agencies
    
    def _clean_party_name(self, name: str) -> str:
        """Clean and normalize party name."""
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Remove trailing punctuation
        name = name.rstrip('.,;:')
        
        # Remove common noise words at the end
        noise_words = ['the', 'a', 'an', 'and', 'or', 'of', 'by', 'for']
        words = name.split()
        while words and words[-1].lower() in noise_words:
            words.pop()
        
        return ' '.join(words)