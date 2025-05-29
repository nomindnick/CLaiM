"""Tests for pattern matching in metadata extraction."""

import pytest
from datetime import datetime

from ..patterns import PatternMatcher


class TestPatternMatcher:
    """Test pattern matching functionality."""
    
    @pytest.fixture
    def matcher(self):
        return PatternMatcher()
    
    def test_extract_dates_us_format(self, matcher):
        """Test extraction of US format dates."""
        text = """
        Project Start Date: 05/15/2024
        RFI submitted on 6/1/2024
        Due date is 12-25-2024
        """
        dates = matcher.extract_dates(text)
        
        assert len(dates) == 3
        assert datetime(2024, 5, 15) in dates
        assert datetime(2024, 6, 1) in dates
        assert datetime(2024, 12, 25) in dates
    
    def test_extract_dates_text_format(self, matcher):
        """Test extraction of text format dates."""
        text = """
        Meeting scheduled for January 15, 2024
        Invoice dated Feb 28, 2024
        Deadline: March 1st, 2024
        """
        dates = matcher.extract_dates(text)
        
        assert len(dates) == 3
        assert datetime(2024, 1, 15) in dates
        assert datetime(2024, 2, 28) in dates
        assert datetime(2024, 3, 1) in dates
    
    def test_extract_reference_numbers(self, matcher):
        """Test extraction of reference numbers."""
        text = """
        RE: RFI #123 - Concrete Specifications
        This responds to your RFI 456 dated 5/1/2024
        
        Change Order No. 007 is approved
        Invoice #2024-0005 attached
        
        Project Number: 2024-ABC-001
        Contract #: C-2024-100
        """
        refs = matcher.extract_reference_numbers(text)
        
        assert "RFI" in refs
        assert "123" in refs["RFI"]
        assert "456" in refs["RFI"]
        
        assert "Change Order" in refs
        assert "007" in refs["Change Order"]
        
        assert "Invoice" in refs
        assert "2024-0005" in refs["Invoice"]
        
        assert "Project" in refs
        assert "2024-ABC-001" in refs["Project"]
        
        assert "Contract" in refs
        assert "C-2024-100" in refs["Contract"]
    
    def test_extract_amounts(self, matcher):
        """Test extraction of monetary amounts."""
        text = """
        Change Order Total: $125,456.78
        Additional cost of $5,000.00
        
        Budget increase: USD 50,000
        
        The total contract value is $1,234,567.89
        """
        amounts = matcher.extract_amounts(text)
        
        assert len(amounts) == 4
        assert 1234567.89 in amounts
        assert 125456.78 in amounts
        assert 50000.0 in amounts
        assert 5000.0 in amounts
    
    def test_extract_parties_from_patterns(self, matcher):
        """Test extraction of parties from explicit patterns."""
        text = """
        Owner: XYZ School District
        General Contractor: ABC Construction Company
        Architect: Smith & Associates Architects
        
        From: John Smith <jsmith@email.com>
        To: Jane Doe <jdoe@company.com>
        """
        parties = matcher.extract_parties(text)
        
        # Check that we found the parties
        party_names = [p["name"] for p in parties]
        assert any("XYZ School District" in name for name in party_names)
        assert any("ABC Construction" in name for name in party_names)
        assert any("Smith" in name for name in party_names)
    
    def test_extract_company_names(self, matcher):
        """Test extraction of company names with suffixes."""
        text = """
        Contract between Johnson Brothers Construction Inc. and 
        Pacific Engineering Associates LLC for the project.
        
        Subcontractor: Metro Plumbing Corp.
        Supplier: Industrial Supply Company
        """
        companies = matcher._extract_company_names(text)
        
        assert "Johnson Brothers Construction Inc." in companies
        assert "Pacific Engineering Associates LLC" in companies
        assert "Metro Plumbing Corp." in companies
        assert "Industrial Supply Company" in companies
    
    def test_extract_agency_names(self, matcher):
        """Test extraction of government agency names."""
        text = """
        Los Angeles Unified School District
        California Department of Transportation
        City of San Francisco
        Orange County Water District
        """
        agencies = matcher._extract_agency_names(text)
        
        assert any("School District" in agency for agency in agencies)
        assert any("Department" in agency for agency in agencies)
        assert any("City of" in agency for agency in agencies)
        assert any("Water District" in agency for agency in agencies)
    
    def test_date_filtering(self, matcher):
        """Test that old/future dates are filtered."""
        text = """
        Historical date: 01/01/1985
        Current date: 05/15/2024
        Far future: 12/31/2050
        """
        dates = matcher.extract_dates(text)
        
        # Should only include dates within reasonable range
        assert len(dates) == 1
        assert datetime(2024, 5, 15) in dates
    
    def test_clean_party_name(self, matcher):
        """Test party name cleaning."""
        assert matcher._clean_party_name("ABC Company, Inc.") == "ABC Company, Inc"
        assert matcher._clean_party_name("XYZ Corp  ") == "XYZ Corp"
        assert matcher._clean_party_name("Company Name and") == "Company Name"
        assert matcher._clean_party_name("The Big Company") == "The Big Company"