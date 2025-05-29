"""Tests for entity normalization."""

import pytest

from ..normalizer import EntityNormalizer


class TestEntityNormalizer:
    """Test entity normalization functionality."""
    
    @pytest.fixture
    def normalizer(self):
        return EntityNormalizer()
    
    def test_normalize_party_name_basic(self, normalizer):
        """Test basic party name normalization."""
        assert normalizer.normalize_party_name("ABC CONSTRUCTION") == "ABC Construction"
        assert normalizer.normalize_party_name("xyz corp.") == "XYZ Corporation"
        assert normalizer.normalize_party_name("Smith & Associates") == "Smith and Associates"
        assert normalizer.normalize_party_name("  Extra   Spaces  ") == "Extra Spaces"
    
    def test_normalize_company_suffixes(self, normalizer):
        """Test normalization of company suffixes."""
        assert normalizer.normalize_party_name("ABC Inc.") == "ABC Incorporated"
        assert normalizer.normalize_party_name("XYZ Corp") == "XYZ Corporation"
        assert normalizer.normalize_party_name("123 Co.") == "123 Company"
        assert normalizer.normalize_party_name("Test Ltd") == "Test Limited"
        assert normalizer.normalize_party_name("Law Firm L.L.C.") == "Law Firm LLC"
        assert normalizer.normalize_party_name("Partners L. L. P.") == "Partners LLP"
    
    def test_normalize_industry_terms(self, normalizer):
        """Test normalization of construction industry terms."""
        assert normalizer.normalize_party_name("ABC Const.") == "ABC Construction"
        assert normalizer.normalize_party_name("XYZ Eng.") == "XYZ Engineering"
        assert normalizer.normalize_party_name("Smith Arch.") == "Smith Architects"
        assert normalizer.normalize_party_name("Metro Elec.") == "Metro Electric"
    
    def test_preserve_ampersand_for_law_firms(self, normalizer):
        """Test that & is preserved for law firms."""
        assert normalizer.normalize_party_name("Smith & Jones Law") == "Smith & Jones Law"
        assert normalizer.normalize_party_name("Legal Partners & Associates") == "Legal Partners & Associates"
        # But not for other companies
        assert normalizer.normalize_party_name("Smith & Sons Construction") == "Smith and Sons Construction"
    
    def test_find_similar_parties(self, normalizer):
        """Test finding similar party names."""
        parties = [
            "ABC Construction Inc.",
            "ABC Construction Incorporated",
            "ABC Const. Inc",
            "XYZ Engineering Corp.",
            "XYZ Engineering Corporation",
            "Different Company LLC"
        ]
        
        similar = normalizer.find_similar_parties(parties)
        
        # Should group ABC variations
        assert any("ABC Construction" in canonical for canonical in similar.keys())
        
        # Should group XYZ variations
        assert any("XYZ Engineering" in canonical for canonical in similar.keys())
        
        # Different Company should not be grouped
        assert not any("Different Company" in variations 
                      for variations_list in similar.values() 
                      for variations in variations_list)
    
    def test_is_abbreviation(self, normalizer):
        """Test abbreviation detection."""
        assert normalizer._is_abbreviation("abc", "abc")
        assert normalizer._is_abbreviation("abc", "abcdef")
        assert normalizer._is_abbreviation("const", "construction")
        assert not normalizer._is_abbreviation("xyz", "abc")
        assert not normalizer._is_abbreviation("ab cd", "ab xyz")
    
    def test_calculate_similarity(self, normalizer):
        """Test similarity calculation."""
        # Exact match
        sim = normalizer._calculate_similarity("ABC Company", "ABC Company")
        assert sim > 0.9
        
        # Same base, different suffix
        sim = normalizer._calculate_similarity("ABC Inc", "ABC Corporation")
        assert sim > 0.9
        
        # Similar but not same
        sim = normalizer._calculate_similarity("ABC Construction", "ABD Construction")
        assert 0.7 < sim < 0.9
        
        # Different
        sim = normalizer._calculate_similarity("ABC Company", "XYZ Company")
        assert sim < 0.5
    
    def test_merge_party_info(self, normalizer):
        """Test merging party information."""
        parties = [
            {"name": "ABC Construction Inc.", "email": "info@abc.com"},
            {"name": "ABC Const. Inc", "phone": "555-1234"},
            {"name": "ABC Construction Incorporated", "role": "General Contractor"},
            {"name": "XYZ Corp", "email": "contact@xyz.com"}
        ]
        
        merged = normalizer.merge_party_info(parties)
        
        # Should have 2 merged parties
        assert len(merged) == 2
        
        # Find ABC party
        abc_party = next(p for p in merged if "ABC" in p["name"])
        assert abc_party["email"] == "info@abc.com"
        assert abc_party["phone"] == "555-1234"
        assert abc_party["role"] == "General Contractor"
        assert len(abc_party.get("aliases", [])) >= 1
        
        # Find XYZ party
        xyz_party = next(p for p in merged if "XYZ" in p["name"])
        assert xyz_party["email"] == "contact@xyz.com"