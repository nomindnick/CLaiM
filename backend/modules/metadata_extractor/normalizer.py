"""Entity normalization for consistent party names."""

import re
from typing import Dict, Set, List, Optional
from difflib import SequenceMatcher
from loguru import logger


class EntityNormalizer:
    """Normalizes entity names for consistency."""
    
    def __init__(self):
        self.company_variations = self._load_company_variations()
        self.name_replacements = self._load_name_replacements()
        self.stop_words = {"the", "a", "an", "and", "&", "of", "for"}
    
    def _load_company_variations(self) -> Dict[str, List[str]]:
        """Load common company name variations."""
        return {
            "incorporated": ["inc", "inc.", "incorporated"],
            "corporation": ["corp", "corp.", "corporation"],
            "company": ["co", "co.", "company"],
            "limited": ["ltd", "ltd.", "limited"],
            "associates": ["assoc", "assoc.", "associates", "assocs"],
            "brothers": ["bros", "bros.", "brothers"],
            "construction": ["const", "const.", "construction"],
            "engineering": ["eng", "eng.", "engineering", "engr", "engr."],
            "architects": ["arch", "arch.", "architects", "architecture"],
            "electric": ["elec", "elec.", "electric", "electrical"],
            "mechanical": ["mech", "mech.", "mechanical"],
            "plumbing": ["plumb", "plumb.", "plumbing", "plbg"],
            "heating": ["htg", "htg.", "heating"],
            "air conditioning": ["ac", "a/c", "air conditioning", "air-conditioning"],
            "development": ["dev", "dev.", "development"],
        }
    
    def _load_name_replacements(self) -> Dict[str, str]:
        """Load standard replacements for normalization."""
        replacements = {}
        
        # Build from variations
        for standard, variations in self.company_variations.items():
            for variant in variations:
                replacements[variant.lower()] = standard
        
        # Add more replacements
        replacements.update({
            "&": "and",
            "+": "and",
            "dept": "department",
            "dept.": "department",
            "div": "division",
            "div.": "division",
            "intl": "international",
            "int'l": "international",
            "natl": "national",
            "nat'l": "national",
        })
        
        return replacements
    
    def normalize_party_name(self, name: str) -> str:
        """Normalize a party name for consistency.
        
        Args:
            name: Raw party name
            
        Returns:
            Normalized name
        """
        if not name:
            return ""
        
        # Convert to title case first
        normalized = name.strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Handle ALL CAPS names
        if normalized.isupper() and len(normalized) > 3:
            normalized = normalized.title()
        
        # Apply replacements
        words = normalized.split()
        normalized_words = []
        
        for word in words:
            word_lower = word.lower().rstrip('.,;:')
            
            # Check for replacement
            if word_lower in self.name_replacements:
                replacement = self.name_replacements[word_lower]
                # Maintain original capitalization style
                if word[0].isupper():
                    replacement = replacement.capitalize()
                normalized_words.append(replacement)
            else:
                normalized_words.append(word)
        
        normalized = ' '.join(normalized_words)
        
        # Fix common patterns
        normalized = self._fix_common_patterns(normalized)
        
        # Remove trailing punctuation
        normalized = normalized.rstrip('.,;:')
        
        return normalized
    
    def _fix_common_patterns(self, name: str) -> str:
        """Fix common formatting patterns."""
        # Fix "L. L. C." -> "LLC"
        name = re.sub(r'\bL\.\s*L\.\s*C\.?\b', 'LLC', name)
        
        # Fix "L. L. P." -> "LLP" 
        name = re.sub(r'\bL\.\s*L\.\s*P\.?\b', 'LLP', name)
        
        # Fix multiple spaces
        name = re.sub(r'\s+', ' ', name)
        
        # Fix space before punctuation
        name = re.sub(r'\s+([,.])', r'\1', name)
        
        # Standardize "and" vs "&"
        if " & " in name:
            # Keep & for law firms and partnerships
            if any(word in name.lower() for word in ["law", "attorneys", "partners"]):
                pass
            else:
                name = name.replace(" & ", " and ")
        
        return name
    
    def find_similar_parties(self, parties: List[str], threshold: float = 0.85) -> Dict[str, List[str]]:
        """Find similar party names that might be the same entity.
        
        Args:
            parties: List of party names
            threshold: Similarity threshold (0-1)
            
        Returns:
            Dictionary mapping canonical names to similar variations
        """
        # Normalize all parties first
        normalized_parties = [(p, self.normalize_party_name(p)) for p in parties]
        
        # Group similar names
        groups = {}
        processed = set()
        
        for i, (orig1, norm1) in enumerate(normalized_parties):
            if norm1 in processed:
                continue
            
            # Start a new group
            similar = [orig1]
            processed.add(norm1)
            
            # Find similar names
            for j, (orig2, norm2) in enumerate(normalized_parties[i+1:], i+1):
                if norm2 in processed:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(norm1, norm2)
                
                if similarity >= threshold:
                    similar.append(orig2)
                    processed.add(norm2)
            
            # Use the longest name as canonical (usually most complete)
            canonical = max(similar, key=len)
            if len(similar) > 1:
                groups[self.normalize_party_name(canonical)] = similar
        
        return groups
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names."""
        # Remove common suffixes for comparison
        suffixes = ["inc", "incorporated", "corp", "corporation", "llc", "company", "co"]
        
        def strip_suffix(name):
            words = name.lower().split()
            if words and words[-1] in suffixes:
                return ' '.join(words[:-1])
            return name.lower()
        
        # Compare without suffixes
        base1 = strip_suffix(name1)
        base2 = strip_suffix(name2)
        
        # If bases are the same, high similarity
        if base1 == base2:
            return 0.95
        
        # Check if one is abbreviation of the other
        if self._is_abbreviation(base1, base2) or self._is_abbreviation(base2, base1):
            return 0.9
        
        # Use sequence matcher for general similarity
        return SequenceMatcher(None, base1, base2).ratio()
    
    def _is_abbreviation(self, short: str, long: str) -> bool:
        """Check if short might be an abbreviation of long."""
        short_parts = short.split()
        long_parts = long.split()
        
        if len(short_parts) != len(long_parts):
            return False
        
        for s, l in zip(short_parts, long_parts):
            # Check if s is abbreviation of l
            if not (s == l or (len(s) <= 4 and l.startswith(s))):
                return False
        
        return True
    
    def merge_party_info(self, parties: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Merge information about the same party from different sources.
        
        Args:
            parties: List of party information dictionaries
            
        Returns:
            Merged party list
        """
        # Group by normalized name
        party_groups = {}
        
        for party in parties:
            name = party.get("name", "")
            if not name:
                continue
            
            normalized = self.normalize_party_name(name)
            
            if normalized not in party_groups:
                party_groups[normalized] = {
                    "name": normalized,
                    "original_names": set(),
                    "roles": set(),
                    "emails": set(),
                    "phones": set(),
                    "companies": set(),
                }
            
            group = party_groups[normalized]
            group["original_names"].add(name)
            
            if party.get("role"):
                group["roles"].add(party["role"])
            if party.get("email"):
                group["emails"].add(party["email"])
            if party.get("phone"):
                group["phones"].add(party["phone"])
            if party.get("company"):
                group["companies"].add(party["company"])
        
        # Convert back to list
        merged = []
        for normalized, group in party_groups.items():
            merged_party = {
                "name": normalized,
                "role": ", ".join(sorted(group["roles"])) if group["roles"] else None,
                "email": sorted(group["emails"])[0] if group["emails"] else None,
                "phone": sorted(group["phones"])[0] if group["phones"] else None,
                "company": sorted(group["companies"])[0] if group["companies"] else None,
                "aliases": sorted(group["original_names"] - {normalized})
            }
            
            # Remove None values
            merged_party = {k: v for k, v in merged_party.items() if v}
            merged.append(merged_party)
        
        return merged