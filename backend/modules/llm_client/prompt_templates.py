"""Prompt templates for CLaiM document processing."""

from typing import List, Optional
from enum import Enum


class DocumentCategory(Enum):
    """Document categories for classification."""
    EMAIL = "Email"
    CONTRACT = "Contract Document"
    CHANGE_ORDER = "Change Order"
    PAYMENT_APPLICATION = "Payment Application"
    INSPECTION_REPORT = "Inspection Report"
    PLANS_SPECIFICATIONS = "Plans and Specifications"
    MEETING_MINUTES = "Meeting Minutes"
    RFI = "Request for Information (RFI)"
    SUBMITTAL = "Submittal"
    DAILY_REPORT = "Daily Report"
    INVOICE = "Invoice"
    LETTER = "Letter"
    OTHER = "Other"


class PromptTemplates:
    """Templates for LLM prompts used in document processing."""
    
    CLASSIFICATION_SYSTEM_PROMPT = """You are an expert legal document classifier specializing in construction law. Your task is to accurately classify the provided document content into one of the following predefined categories."""
    
    CLASSIFICATION_USER_TEMPLATE = """Please classify the document content below into one of these categories:
{categories}

Document Content:
---
{document_text}
---

{context_section}

Respond with only the category name, followed by a confidence score (0-100), followed by a brief reason.
Format: CATEGORY | CONFIDENCE | REASON

Classification:"""
    
    BOUNDARY_SYSTEM_PROMPT = """You are an expert in analyzing document structures within large concatenated texts. Your task is to determine if the 'Next Text Block' starts a new logical document or continues the 'Current Document Excerpt'. Consider changes in topic, common document start/end phrases, formatting cues (if any are preserved in the text), and overall coherence."""
    
    BOUNDARY_USER_TEMPLATE = """Current Document Excerpt (last ~150 words):
---
{current_segment_end}
---

Next Text Block (first ~150 words):
---
{next_segment_start}
---

Does the 'Next Text Block' appear to start a NEW document, distinct from the 'Current Document Excerpt'? Consider:
- Topic changes
- Document headers/footers
- Sender/recipient changes
- Date/time discontinuities
- Format changes
- Signature blocks

Answer with only YES or NO, followed by confidence (0-100), followed by the primary reason.
Format: YES/NO | CONFIDENCE | REASON

Answer:"""
    
    @classmethod
    def format_classification_prompt(cls, 
                                   document_text: str,
                                   categories: Optional[List[str]] = None,
                                   context: Optional[str] = None) -> str:
        """Format classification prompt with document text.
        
        Args:
            document_text: Text content to classify
            categories: List of category names (uses default if None)
            context: Optional context like title or filename
            
        Returns:
            Formatted prompt string
        """
        # Use default categories if none provided
        if categories is None:
            categories = [cat.value for cat in DocumentCategory]
        
        # Format categories as bullet points
        category_list = "\n".join(f"- {cat}" for cat in categories)
        
        # Add context section if provided
        context_section = ""
        if context:
            context_section = f"\nAdditional Context: {context}\n"
        
        # Truncate document text if too long (keep first and last parts)
        max_length = 4000
        if len(document_text) > max_length:
            truncate_point = max_length // 2
            document_text = (
                document_text[:truncate_point] + 
                "\n\n[... content truncated ...]\n\n" + 
                document_text[-truncate_point:]
            )
        
        return cls.CLASSIFICATION_USER_TEMPLATE.format(
            categories=category_list,
            document_text=document_text,
            context_section=context_section
        )
    
    @classmethod
    def format_boundary_prompt(cls,
                             current_segment_end: str,
                             next_segment_start: str) -> str:
        """Format boundary detection prompt.
        
        Args:
            current_segment_end: End portion of current document
            next_segment_start: Start portion of next potential document
            
        Returns:
            Formatted prompt string
        """
        # Truncate segments to ~150 words each
        current_words = current_segment_end.split()
        next_words = next_segment_start.split()
        
        if len(current_words) > 150:
            current_segment_end = " ".join(current_words[-150:])
        
        if len(next_words) > 150:
            next_segment_start = " ".join(next_words[:150])
        
        return cls.BOUNDARY_USER_TEMPLATE.format(
            current_segment_end=current_segment_end,
            next_segment_start=next_segment_start
        )
    
    @classmethod
    def parse_classification_response(cls, response: str) -> dict:
        """Parse classification response into structured data.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Dictionary with parsed classification data
        """
        try:
            # Expected format: CATEGORY | CONFIDENCE | REASON
            parts = response.strip().split(" | ", 2)
            
            if len(parts) >= 2:
                category = parts[0].strip()
                confidence_str = parts[1].strip()
                reason = parts[2].strip() if len(parts) > 2 else "No reason provided"
                
                # Parse confidence
                try:
                    confidence = float(confidence_str) / 100.0  # Convert to 0-1 range
                except ValueError:
                    confidence = 0.0
                
                return {
                    "document_type": category,
                    "confidence": confidence,
                    "reasoning": reason,
                    "raw_response": response
                }
            else:
                # Fallback parsing
                return {
                    "document_type": "Other",
                    "confidence": 0.0,
                    "reasoning": f"Could not parse response: {response}",
                    "raw_response": response
                }
                
        except Exception as e:
            return {
                "document_type": "Other",
                "confidence": 0.0,
                "reasoning": f"Parse error: {str(e)}",
                "raw_response": response
            }
    
    @classmethod
    def parse_boundary_response(cls, response: str) -> dict:
        """Parse boundary detection response into structured data.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Dictionary with parsed boundary data
        """
        try:
            # Expected format: YES/NO | CONFIDENCE | REASON
            parts = response.strip().split(" | ", 2)
            
            if len(parts) >= 2:
                decision = parts[0].strip().upper()
                confidence_str = parts[1].strip()
                reason = parts[2].strip() if len(parts) > 2 else "No reason provided"
                
                # Parse decision
                is_boundary = decision == "YES"
                
                # Parse confidence
                try:
                    confidence = float(confidence_str) / 100.0  # Convert to 0-1 range
                except ValueError:
                    confidence = 0.0
                
                return {
                    "is_boundary": is_boundary,
                    "confidence": confidence,
                    "reasoning": reason,
                    "raw_response": response
                }
            else:
                # Fallback parsing
                return {
                    "is_boundary": False,
                    "confidence": 0.0,
                    "reasoning": f"Could not parse response: {response}",
                    "raw_response": response
                }
                
        except Exception as e:
            return {
                "is_boundary": False,
                "confidence": 0.0,
                "reasoning": f"Parse error: {str(e)}",
                "raw_response": response
            }
    
    @classmethod
    def get_test_prompts(cls) -> dict:
        """Get test prompts for validation.
        
        Returns:
            Dictionary with test prompts for different tasks
        """
        return {
            "classification": {
                "email": cls.format_classification_prompt(
                    "From: john@contractor.com\nTo: susan@owner.com\nSubject: RE: Project Update\n\nDear Susan,\n\nThank you for your email regarding the foundation work. We have completed the excavation and will begin pouring concrete tomorrow morning. Please let me know if you have any questions.\n\nBest regards,\nJohn Smith"
                ),
                "rfi": cls.format_classification_prompt(
                    "REQUEST FOR INFORMATION\nRFI #045\nDate: March 15, 2024\nProject: School Construction\n\nQuestion: The architectural drawings show a 4-inch water line in conflict with the structural beam detail on drawing S-3. Please clarify the routing of the water line.\n\nResponse Required By: March 20, 2024"
                ),
                "invoice": cls.format_classification_prompt(
                    "INVOICE #INV-2024-0892\nABC Construction Company\n123 Main Street\n\nBill To: School District\nDate: March 15, 2024\nDue Date: April 15, 2024\n\nDescription: Concrete work - Foundation\nQuantity: 250 cubic yards\nRate: $150.00\nAmount: $37,500.00\n\nTotal Amount Due: $37,500.00"
                )
            },
            "boundary": {
                "clear_boundary": cls.format_boundary_prompt(
                    "Best regards,\nJohn Smith\nProject Manager\nABC Construction\n\n[End of Email]",
                    "REQUEST FOR INFORMATION\nRFI #045\nDate: March 15, 2024\nProject: Elementary School\n\nTo: Design Team\nFrom: General Contractor"
                ),
                "continuation": cls.format_boundary_prompt(
                    "The foundation work is proceeding on schedule. We encountered some unexpected soil conditions in the northeast corner that required additional excavation.",
                    "The additional excavation added approximately 2 days to the schedule, but we were able to maintain the overall project timeline by adjusting the crew assignments for the following week."
                )
            }
        }