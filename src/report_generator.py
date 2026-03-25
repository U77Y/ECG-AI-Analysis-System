"""
Report Generation Module - Enhanced Version
With Doctor Recommendations, Patient Explanations, and Full References
"""

import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from src.clinical_advisor import ClinicalAdvisor

class ReportGenerator:
    """Class ya kutengeneza reports za ECG analysis with clinical insights"""
    
    def __init__(self):
        self.report_data = {}
        self.clinical_advisor = ClinicalAdvisor()
        self.accuracy_target = 0.95  # 95% accuracy target
        
    def add_analysis_result(self, result, heart_rate, r_peaks_count, patient_info=None):
        """
        Add ECG analysis results to report with clinical interpretation
        """
        # Get doctor recommendations
        doctor_advice = self.clinical_advisor.get_doctor_recommendations(result, heart_rate)
        
        # Get patient explanation
        patient_explanation = self.clinical_advisor.get_patient_explanation(result, heart_rate)
        
        self.report_data = {
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'condition': result['condition'],
            'confidence': result['confidence'],
            'heart_rate': heart_rate,
            'r_peaks_count': r_peaks_count,
            'doctor_recommendations': doctor_advice,
            'patient_explanation': patient_explanation,
            'accuracy_achieved': result['confidence'],
            'accuracy_target': self.accuracy_target,
            'ai_confidence_statement': self._get_confidence_statement(result['confidence'])
        }
        
    def _get_confidence_statement(self, confidence):
        """Generate AI confidence statement"""
        if confidence >= 0.95:
            return f"✅ HIGH CONFIDENCE: Analysis accuracy {confidence:.1%} - Meeting 95% target"
        elif confidence >= 0.85:
            return f"⚠️ GOOD CONFIDENCE: Analysis accuracy {confidence:.1%} - Approaching 95% target"
        else:
            return f"⚠️ MODERATE CONFIDENCE: Analysis accuracy {confidence:.1%} - Clinical correlation recommended"
            
    def generate_pdf_report(self, filename="ecg_report.pdf", plot_image=None):
        """
        Generate comprehensive PDF report with clinical recommendations
        """
        doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=72, leftMargin=72)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, 
                                     alignment=TA_CENTER, spaceAfter=30, textColor=colors.HexColor('#2E86AB'))
        
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14,
                                       textColor=colors.HexColor('#1F5E3A'), spaceAfter=12)
        
        clinical_style = ParagraphStyle('ClinicalStyle', parent=styles['Normal'], fontSize=11,
                                        alignment=TA_JUSTIFY, spaceAfter=6)
        
        # Title
        title = Paragraph("ECG Clinical Analysis Report", title_style)
        story.append(title)
        
        # Date and AI Info
        date_style = ParagraphStyle('DateStyle', parent=styles['Normal'], alignment=TA_CENTER, fontSize=10)
        date_info = Paragraph(f"Generated: {self.report_data['analysis_date']}<br/>"
                             f"AI System: ECG Analysis v2.0 | Target Accuracy: 95% | Achieved: {self.report_data['accuracy_achieved']:.1%}", date_style)
        story.append(date_info)
        story.append(Spacer(1, 20))
        
        # Patient Summary Section
        story.append(Paragraph("PATIENT SUMMARY", heading_style))
        story.append(Spacer(1, 6))
        
        summary_data = [
            ["Finding", "Value", "Clinical Significance"],
            ["Heart Rate", f"{self.report_data['heart_rate']:.1f} bpm", 
             "Normal" if 60 <= self.report_data['heart_rate'] <= 100 else "Abnormal"],
            ["ECG Pattern", self.report_data['condition'], 
             "Regular" if self.report_data['condition'] == 'Normal' else "Requires Review"],
            ["AI Confidence", f"{self.report_data['accuracy_achieved']:.1%}", 
             self.report_data['ai_confidence_statement'].split('-')[0]]
        ]
        
        table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Section: What This Means for the Patient
        story.append(Paragraph("WHAT THIS MEANS FOR YOU", heading_style))
        story.append(Spacer(1, 6))
        patient_text = Paragraph(f"{self.report_data['patient_explanation']['what_it_means']}<br/><br/>"
                                f"{self.report_data['patient_explanation']['condition_explained']}", clinical_style)
        story.append(patient_text)
        story.append(Spacer(1, 10))
        
        # Lifestyle Advice
        if self.report_data['patient_explanation']['lifestyle_advice']:
            story.append(Paragraph("Recommended Lifestyle Actions:", ParagraphStyle('ListHead', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold')))
            for advice in self.report_data['patient_explanation']['lifestyle_advice']:
                story.append(Paragraph(f"• {advice}", clinical_style))
        story.append(Spacer(1, 15))
        
        # Section: Doctor's Clinical Recommendations
        story.append(PageBreak())
        story.append(Paragraph("CLINICAL RECOMMENDATIONS FOR HEALTHCARE PROVIDER", heading_style))
        story.append(Spacer(1, 6))
        
        doc_advice = self.report_data['doctor_recommendations']
        story.append(Paragraph(f"<b>Summary:</b> {doc_advice['summary']}", clinical_style))
        story.append(Spacer(1, 8))
        
        story.append(Paragraph("<b>Clinical Interpretation:</b>", ParagraphStyle('Bold', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold')))
        story.append(Paragraph(doc_advice['clinical_interpretation'], clinical_style))
        story.append(Spacer(1, 8))
        
        if doc_advice['recommended_actions']:
            story.append(Paragraph("<b>Recommended Actions:</b>", ParagraphStyle('Bold', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold')))
            for action in doc_advice['recommended_actions']:
                story.append(Paragraph(f"• {action}", clinical_style))
        story.append(Spacer(1, 8))
        
        if doc_advice['referral_needed']:
            story.append(Paragraph("<b>⚠️ Referral Recommended:</b> Cardiology consultation advised.", 
                                  ParagraphStyle('Warning', parent=styles['Normal'], fontSize=11, textColor=colors.red)))
            story.append(Spacer(1, 8))
        
        # Add plot if available
        if plot_image and os.path.exists(plot_image):
            story.append(Spacer(1, 10))
            story.append(Paragraph("ECG WAVEFORM", heading_style))
            img = Image(plot_image, width=6*inch, height=3*inch)
            story.append(img)
        
        # Full WHO References
        story.append(PageBreak())
        story.append(Paragraph("WORLD HEALTH ORGANIZATION (WHO) REFERENCES", heading_style))
        story.append(Spacer(1, 6))
        
        who_refs = self.clinical_advisor.get_full_who_references()
        for line in who_refs.split('\n'):
            if line.strip():
                story.append(Paragraph(line, clinical_style))
                story.append(Spacer(1, 3))
        
        # Footer
        story.append(Spacer(1, 30))
        footer = Paragraph("This report is generated by ECG AI Analysis System v2.0 (95% Accuracy Target). "
                          "Clinical decisions should be made by qualified healthcare professionals. "
                          "AI analysis should be used as a supportive tool, not a substitute for clinical judgment.", 
                          ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER))
        story.append(footer)
        
        doc.build(story)
        print(f"✅ PDF report generated: {filename}")
        return filename
        
    def generate_docx_report(self, filename="ecg_report.docx", plot_image=None):
        """
        Generate comprehensive Word report
        """
        doc = Document()
        
        # Title
        title = doc.add_heading('ECG Clinical Analysis Report', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Date and AI Info
        doc.add_paragraph(f"Generated: {self.report_data['analysis_date']}")
        doc.add_paragraph(f"AI System: ECG Analysis v2.0 | Target Accuracy: 95% | Achieved: {self.report_data['accuracy_achieved']:.1%}")
        doc.add_paragraph()
        
        # Patient Summary
        doc.add_heading('PATIENT SUMMARY', level=1)
        table = doc.add_table(rows=4, cols=3)
        table.style = 'Table Grid'
        
        headers = table.rows[0].cells
        headers[0].text = 'Finding'
        headers[1].text = 'Value'
        headers[2].text = 'Clinical Significance'
        
        rows_data = [
            ('Heart Rate', f"{self.report_data['heart_rate']:.1f} bpm", 
             "Normal" if 60 <= self.report_data['heart_rate'] <= 100 else "Abnormal"),
            ('ECG Pattern', self.report_data['condition'], 
             "Regular" if self.report_data['condition'] == 'Normal' else "Requires Review"),
            ('AI Confidence', f"{self.report_data['accuracy_achieved']:.1%}", 
             self.report_data['ai_confidence_statement'])
        ]
        
        for i, (param, value, signif) in enumerate(rows_data, 1):
            row = table.rows[i].cells
            row[0].text = param
            row[1].text = value
            row[2].text = signif
            
        doc.add_paragraph()
        
        # Patient Explanation
        doc.add_heading('WHAT THIS MEANS FOR YOU', level=1)
        patient = self.report_data['patient_explanation']
        doc.add_paragraph(patient['what_it_means'])
        doc.add_paragraph(patient['condition_explained'])
        
        if patient['lifestyle_advice']:
            doc.add_heading('Recommended Lifestyle Actions', level=2)
            for advice in patient['lifestyle_advice']:
                doc.add_paragraph(advice, style='List Bullet')
                
        if patient['when_to_seek_help']:
            doc.add_heading('When to Seek Medical Help', level=2)
            for help_advice in patient['when_to_seek_help']:
                doc.add_paragraph(help_advice, style='List Bullet')
                
        doc.add_paragraph()
        
        # Doctor Recommendations
        doc.add_heading('CLINICAL RECOMMENDATIONS FOR HEALTHCARE PROVIDER', level=1)
        doc_advice = self.report_data['doctor_recommendations']
        doc.add_paragraph(f"Summary: {doc_advice['summary']}")
        doc.add_paragraph(f"Clinical Interpretation: {doc_advice['clinical_interpretation']}")
        
        if doc_advice['recommended_actions']:
            doc.add_heading('Recommended Actions', level=2)
            for action in doc_advice['recommended_actions']:
                doc.add_paragraph(action, style='List Bullet')
                
        if doc_advice['referral_needed']:
            doc.add_paragraph("⚠️ Referral Recommended: Cardiology consultation advised")
            
        # Add plot
        if plot_image and os.path.exists(plot_image):
            doc.add_heading('ECG WAVEFORM', level=1)
            doc.add_picture(plot_image, width=Inches(6))
            
        # WHO References
        doc.add_page_break()
        doc.add_heading('WORLD HEALTH ORGANIZATION (WHO) REFERENCES', level=1)
        
        who_refs = self.clinical_advisor.get_full_who_references()
        for line in who_refs.split('\n'):
            if line.strip():
                doc.add_paragraph(line)
                
        # Save
        doc.save(filename)
        print(f"✅ Word report generated: {filename}")
        return filename
