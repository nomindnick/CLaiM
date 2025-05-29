#!/usr/bin/env python3
"""Generate test construction PDFs for CLaiM testing.

Creates realistic construction litigation documents including:
- RFIs (Request for Information)
- Change Orders
- Daily Reports
- Invoices
- Contracts
- Emails
- Mixed text/scanned pages
"""

from pathlib import Path
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont
import random
import textwrap


class TestPDFGenerator:
    """Generate realistic construction document PDFs for testing."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.styles = getSampleStyleSheet()
        
        # Add custom styles for construction documents
        self.styles.add(ParagraphStyle(
            name='DocumentHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            alignment=1  # Center
        ))
        
        self.styles.add(ParagraphStyle(
            name='FieldLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey
        ))
    
    def generate_rfi(self, rfi_number: int = 123) -> Path:
        """Generate a Request for Information (RFI) document."""
        filename = self.output_dir / f"RFI_{rfi_number:03d}.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        story = []
        
        # Header
        story.append(Paragraph("REQUEST FOR INFORMATION", self.styles['DocumentHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        # RFI Details Table
        rfi_data = [
            ['RFI Number:', f'RFI-{rfi_number:03d}', 'Date:', datetime.now().strftime('%B %d, %Y')],
            ['Project:', 'Lincoln Elementary School Modernization', 'Project No:', '2023-EDU-042'],
            ['To:', 'ABC Construction, Inc.', 'Attn:', 'John Smith, Project Manager'],
            ['From:', 'XYZ School District', 'Submitted By:', 'Jane Doe, Facilities Director'],
        ]
        
        t = Table(rfi_data, colWidths=[1.5*inch, 2.5*inch, 1*inch, 2*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Subject
        story.append(Paragraph("<b>SUBJECT:</b> Concrete Strength Requirements - Gymnasium Foundation", 
                             self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Question
        story.append(Paragraph("<b>QUESTION:</b>", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        question_text = """
        The structural drawings specify 4,000 PSI concrete for the gymnasium foundation. However, 
        the geotechnical report dated January 15, 2024, indicates soil conditions that may require 
        higher strength concrete. Specifically, the report notes:
        
        1. High groundwater table at -8 feet below grade
        2. Presence of expansive soils with plasticity index of 35
        3. Seismic design category D requirements
        
        Please clarify if the specified 4,000 PSI concrete strength is adequate given these 
        conditions, or if a higher strength mix design should be used. If a different mix is 
        required, please provide updated specifications including:
        
        - Minimum compressive strength at 28 days
        - Water-cement ratio requirements
        - Admixture requirements for waterproofing
        - Any special curing procedures
        """
        
        for line in question_text.strip().split('\n'):
            if line.strip():
                story.append(Paragraph(line, self.styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Response required by
        story.append(Paragraph(f"<b>RESPONSE REQUIRED BY:</b> {(datetime.now() + timedelta(days=5)).strftime('%B %d, %Y')}", 
                             self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        # Impact
        story.append(Paragraph("<b>SCHEDULE IMPACT:</b> Critical - Foundation pour scheduled for next week", 
                             self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<b>COST IMPACT:</b> Potential change order if mix design changes required", 
                             self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return filename
    
    def generate_change_order(self, co_number: int = 7) -> Path:
        """Generate a Change Order document."""
        filename = self.output_dir / f"Change_Order_{co_number:03d}.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        story = []
        
        # Header
        story.append(Paragraph("CHANGE ORDER", self.styles['DocumentHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        # Change Order Details
        co_data = [
            ['Change Order No:', f'CO-{co_number:03d}', 'Date:', datetime.now().strftime('%B %d, %Y')],
            ['Project:', 'Lincoln Elementary School Modernization', 'Contract:', 'C-2023-042'],
            ['Owner:', 'XYZ School District', 'Contractor:', 'ABC Construction, Inc.'],
            ['Original Contract Sum:', '$8,543,200.00', 'Previous Changes:', '$125,430.00'],
        ]
        
        t = Table(co_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Description of Change
        story.append(Paragraph("<b>DESCRIPTION OF CHANGE:</b>", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        change_items = [
            "1. Upgrade gymnasium foundation concrete from 4,000 PSI to 5,000 PSI per RFI-123",
            "2. Add waterproofing admixture to concrete mix design",
            "3. Extend curing period from 7 days to 14 days with moisture retention",
            "4. Additional reinforcement steel (#5 rebar at 12\" O.C. each way)",
        ]
        
        for item in change_items:
            story.append(Paragraph(item, self.styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Cost Breakdown
        story.append(Paragraph("<b>COST BREAKDOWN:</b>", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        cost_data = [
            ['Item', 'Quantity', 'Unit Price', 'Total'],
            ['5,000 PSI Concrete', '850 CY', '$25.00/CY', '$21,250.00'],
            ['Waterproofing Admixture', '850 CY', '$8.50/CY', '$7,225.00'],
            ['Additional Labor - Curing', '120 hrs', '$85.00/hr', '$10,200.00'],
            ['Additional Rebar', '18,500 lbs', '$0.95/lb', '$17,575.00'],
            ['', '', 'Subtotal:', '$56,250.00'],
            ['', '', 'Overhead & Profit (15%):', '$8,437.50'],
            ['', '', '<b>Total This Change Order:</b>', '<b>$64,687.50</b>'],
        ]
        
        t = Table(cost_data, colWidths=[3*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (1, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (-2, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
            ('LINEABOVE', (2, -3), (-1, -3), 0.5, colors.grey),
            ('LINEABOVE', (2, -1), (-1, -1), 1, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Schedule Impact
        story.append(Paragraph("<b>SCHEDULE IMPACT:</b> 7 calendar days extension for extended curing period", 
                             self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return filename
    
    def generate_daily_report(self, report_date: datetime = None) -> Path:
        """Generate a Daily Report document."""
        if report_date is None:
            report_date = datetime.now() - timedelta(days=random.randint(1, 30))
        
        filename = self.output_dir / f"Daily_Report_{report_date.strftime('%Y%m%d')}.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        story = []
        
        # Header
        story.append(Paragraph("DAILY CONSTRUCTION REPORT", self.styles['DocumentHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        # Report Details
        report_data = [
            ['Date:', report_date.strftime('%B %d, %Y'), 'Report No:', f'DR-{report_date.strftime("%Y%m%d")}'],
            ['Project:', 'Lincoln Elementary School Modernization', 'Weather:', 'Clear, 72°F'],
            ['Contractor:', 'ABC Construction, Inc.', 'Superintendent:', 'Mike Johnson'],
        ]
        
        t = Table(report_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Work Performed
        story.append(Paragraph("<b>WORK PERFORMED:</b>", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        work_items = [
            "• Continued excavation for gymnasium foundation - 75% complete",
            "• Installed formwork for north wall foundation",
            "• Placed reinforcing steel for footings F-12 through F-18",
            "• Concrete pour for footings F-1 through F-6 (185 CY)",
            "• Backfilled and compacted areas around completed footings",
        ]
        
        for item in work_items:
            story.append(Paragraph(item, self.styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Manpower
        story.append(Paragraph("<b>MANPOWER ON SITE:</b>", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        manpower_data = [
            ['Trade', 'Count', 'Trade', 'Count'],
            ['Laborers', '12', 'Concrete Finishers', '6'],
            ['Carpenters', '8', 'Equipment Operators', '4'],
            ['Iron Workers', '10', 'Foreman/Supervisors', '3'],
        ]
        
        t = Table(manpower_data, colWidths=[2*inch, 1*inch, 2*inch, 1*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ALIGN', (3, 0), (3, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*inch))
        
        # Issues/Delays
        story.append(Paragraph("<b>ISSUES/DELAYS:</b>", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("• Waiting for response to RFI-123 regarding concrete strength requirements", 
                             self.styles['Normal']))
        story.append(Paragraph("• Minor delay (2 hours) due to concrete truck breakdown", 
                             self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return filename
    
    def generate_invoice(self, invoice_number: int = 5) -> Path:
        """Generate an Invoice document."""
        filename = self.output_dir / f"Invoice_{invoice_number:04d}.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        story = []
        
        # Header
        story.append(Paragraph("INVOICE", self.styles['DocumentHeader']))
        story.append(Spacer(1, 0.2*inch))
        
        # Invoice Details
        invoice_data = [
            ['Invoice No:', f'INV-{invoice_number:04d}', 'Date:', datetime.now().strftime('%B %d, %Y')],
            ['Project:', 'Lincoln Elementary School Modernization', 'Contract:', 'C-2023-042'],
            ['Bill To:', 'XYZ School District', 'Period:', f'{(datetime.now() - timedelta(days=30)).strftime("%B %Y")}'],
            ['', '123 Education Blvd', '', ''],
            ['', 'Springfield, CA 90210', '', ''],
        ]
        
        t = Table(invoice_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, 2), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, 2), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('SPAN', (1, 3), (1, 4)),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Schedule of Values
        story.append(Paragraph("<b>SCHEDULE OF VALUES - PROGRESS BILLING</b>", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        sov_data = [
            ['Item', 'Description', 'Contract Value', 'Previous', 'This Period', 'Total', '% Complete'],
            ['1', 'Mobilization', '$85,432.00', '$85,432.00', '$0.00', '$85,432.00', '100%'],
            ['2', 'Site Preparation', '$342,150.00', '$273,720.00', '$68,430.00', '$342,150.00', '100%'],
            ['3', 'Foundation Work', '$856,320.00', '$428,160.00', '$214,080.00', '$642,240.00', '75%'],
            ['4', 'Structural Steel', '$1,284,480.00', '$0.00', '$128,448.00', '$128,448.00', '10%'],
            ['5', 'Concrete Work', '$742,560.00', '$148,512.00', '$185,640.00', '$334,152.00', '45%'],
            ['', '', '', '', '', '', ''],
            ['', '<b>Subtotal</b>', '', '', '<b>$596,598.00</b>', '', ''],
            ['', 'Retention (10%)', '', '', '($59,659.80)', '', ''],
            ['', '<b>Total Due This Period</b>', '', '', '<b>$536,938.20</b>', '', ''],
        ]
        
        t = Table(sov_data, colWidths=[0.5*inch, 2.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 0.8*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (1, -3), (1, -1), 'Helvetica-Bold'),
            ('FONTNAME', (4, -3), (4, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
            ('LINEABOVE', (3, -3), (4, -3), 0.5, colors.grey),
            ('LINEABOVE', (3, -1), (4, -1), 1, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        
        # Build PDF
        doc.build(story)
        return filename
    
    def generate_email(self) -> Path:
        """Generate an Email document (as PDF)."""
        filename = self.output_dir / f"Email_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(str(filename), pagesize=letter)
        story = []
        
        # Email Header
        email_style = ParagraphStyle(
            name='EmailHeader',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            spaceAfter=6
        )
        
        story.append(Paragraph("<b>From:</b> john.smith@abcconstruction.com", email_style))
        story.append(Paragraph("<b>To:</b> jane.doe@xyzschooldistrict.edu", email_style))
        story.append(Paragraph("<b>Date:</b> " + datetime.now().strftime('%B %d, %Y at %I:%M %p'), email_style))
        story.append(Paragraph("<b>Subject:</b> RE: Gymnasium Foundation - Concrete Mix Design", email_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Email Body
        story.append(Paragraph("Jane,", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        email_body = """
        Following up on our discussion yesterday regarding the gymnasium foundation concrete mix design.
        
        After reviewing the geotechnical report and consulting with our structural engineer, we recommend 
        the following modifications to the original specifications:
        
        1. Increase concrete strength from 4,000 PSI to 5,000 PSI
        2. Add crystalline waterproofing admixture at 2% by weight of cement
        3. Maximum water-cement ratio of 0.45
        4. Extend wet curing period to 14 days minimum
        
        These changes will address the concerns raised in RFI-123 regarding the high water table and 
        expansive soil conditions. The additional cost for these modifications is estimated at $64,687.50, 
        which we will submit as Change Order #7 for your approval.
        
        Please let me know if you need any additional information or if you'd like to schedule a meeting 
        to discuss this further.
        
        Best regards,
        John Smith
        Project Manager
        ABC Construction, Inc.
        (555) 123-4567
        """
        
        for paragraph in email_body.strip().split('\n\n'):
            story.append(Paragraph(paragraph.strip(), self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(story)
        return filename
    
    def generate_mixed_document(self) -> Path:
        """Generate a document with mixed text and scanned pages."""
        filename = self.output_dir / "Mixed_Document_Contract_Amendment.pdf"
        
        # Create a canvas for direct drawing
        c = canvas.Canvas(str(filename), pagesize=letter)
        width, height = letter
        
        # Page 1: Normal text page (Contract Amendment header)
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width/2, height - inch, "CONTRACT AMENDMENT NO. 3")
        
        c.setFont("Helvetica", 12)
        y = height - 2*inch
        
        lines = [
            "Project: Lincoln Elementary School Modernization",
            "Contract No: C-2023-042",
            "Date: " + datetime.now().strftime('%B %d, %Y'),
            "",
            "This Amendment modifies the original contract dated January 15, 2024,",
            "between XYZ School District (Owner) and ABC Construction, Inc. (Contractor).",
            "",
            "WHEREAS, unforeseen site conditions have been discovered requiring",
            "modifications to the foundation design and construction methods;",
            "",
            "WHEREAS, both parties agree that these modifications are necessary",
            "for the successful completion of the project;",
            "",
            "NOW, THEREFORE, the parties agree to the following amendments:",
        ]
        
        for line in lines:
            c.drawString(inch, y, line)
            y -= 20
        
        c.showPage()
        
        # Page 2: Simulated scanned page (slightly rotated, with noise)
        # Create an image to simulate a scanned page
        img_width, img_height = 612, 792  # Letter size in points
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Add some noise to simulate scan artifacts
        for _ in range(1000):
            x = random.randint(0, img_width)
            y = random.randint(0, img_height)
            gray = random.randint(200, 255)
            draw.point((x, y), fill=(gray, gray, gray))
        
        # Add text to the image (simulating scanned text)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 14)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
            font_bold = font
        
        # Slightly rotate the text to simulate scan skew
        text_img = Image.new('RGB', (img_width, img_height), 'white')
        text_draw = ImageDraw.Draw(text_img)
        
        y = 100
        text_draw.text((80, y), "AMENDMENT DETAILS", font=font_bold, fill='black')
        y += 40
        
        details = [
            "1. Foundation Design Modifications:",
            "   - Increase footing depth from 4'-0\" to 6'-0\"",
            "   - Add grade beams between all column footings",
            "   - Upgrade concrete to 5,000 PSI throughout",
            "",
            "2. Additional Scope of Work:",
            "   - Install French drain system around gymnasium",
            "   - Waterproof all below-grade concrete",
            "   - Add vapor barrier under slab-on-grade",
            "",
            "3. Contract Sum Adjustment:",
            "   Original Contract Sum:        $8,543,200.00",
            "   Previous Amendments:          $  189,117.50",
            "   This Amendment:              $  156,825.00",
            "   New Contract Sum:            $8,889,142.50",
        ]
        
        for line in details:
            text_draw.text((80, y), line, font=font, fill='black')
            y += 25
        
        # Rotate slightly to simulate scan skew
        text_img = text_img.rotate(-0.5, fillcolor='white')
        
        # Composite the text onto the noisy background
        img.paste(text_img, (0, 0))
        
        # Save the image temporarily
        temp_img_path = self.output_dir / "temp_scanned_page.png"
        img.save(temp_img_path)
        
        # Add the image to the PDF
        c.drawImage(str(temp_img_path), 0, 0, width, height)
        c.showPage()
        
        # Page 3: Normal text page (signatures)
        c.setFont("Helvetica", 12)
        y = height - 2*inch
        
        c.drawString(inch, y, "IN WITNESS WHEREOF, the parties have executed this Amendment")
        y -= 20
        c.drawString(inch, y, "as of the date first written above.")
        y -= 60
        
        # Signature blocks
        c.drawString(inch, y, "OWNER:")
        c.drawString(width/2 + inch/2, y, "CONTRACTOR:")
        y -= 40
        
        c.drawString(inch, y, "_" * 40)
        c.drawString(width/2 + inch/2, y, "_" * 40)
        y -= 20
        
        c.drawString(inch, y, "Jane Doe")
        c.drawString(width/2 + inch/2, y, "John Smith")
        y -= 20
        
        c.drawString(inch, y, "Facilities Director")
        c.drawString(width/2 + inch/2, y, "Project Manager")
        y -= 20
        
        c.drawString(inch, y, "XYZ School District")
        c.drawString(width/2 + inch/2, y, "ABC Construction, Inc.")
        
        # Save the PDF
        c.save()
        
        # Clean up temporary image
        temp_img_path.unlink()
        
        return filename
    
    def generate_all_test_documents(self) -> list[Path]:
        """Generate a complete set of test documents."""
        generated_files = []
        
        print("Generating test construction documents...")
        
        # Generate multiple RFIs
        for i in [123, 124, 125]:
            file = self.generate_rfi(i)
            generated_files.append(file)
            print(f"  ✓ Generated {file.name}")
        
        # Generate change orders
        for i in [7, 8]:
            file = self.generate_change_order(i)
            generated_files.append(file)
            print(f"  ✓ Generated {file.name}")
        
        # Generate daily reports
        for i in range(3):
            file = self.generate_daily_report()
            generated_files.append(file)
            print(f"  ✓ Generated {file.name}")
        
        # Generate invoices
        for i in [5, 6]:
            file = self.generate_invoice(i)
            generated_files.append(file)
            print(f"  ✓ Generated {file.name}")
        
        # Generate emails
        for i in range(2):
            file = self.generate_email()
            generated_files.append(file)
            print(f"  ✓ Generated {file.name}")
        
        # Generate mixed document
        file = self.generate_mixed_document()
        generated_files.append(file)
        print(f"  ✓ Generated {file.name}")
        
        print(f"\nGenerated {len(generated_files)} test documents in {self.output_dir}")
        return generated_files


if __name__ == "__main__":
    # Generate test documents
    output_dir = Path(__file__).parent
    generator = TestPDFGenerator(output_dir)
    files = generator.generate_all_test_documents()