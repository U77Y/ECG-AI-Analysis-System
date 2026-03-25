"""
Email Sending Module
Tuma reports kwa email
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

class EmailSender:
    """Class ya kutuma reports kwa email"""
    
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
    def send_report(self, to_email, report_files, patient_name="Patient", doctor_email=None):
        """
        Tuma report kwa email
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = "ecg.analysis.system@gmail.com"  # Replace with your email
            msg['To'] = to_email
            msg['Subject'] = f"ECG Analysis Report - {patient_name}"
            
            # Email body
            body = f"""
            Dear {patient_name},
            
            Your ECG analysis report is attached.
            
            Analysis Summary:
            - Heart Rate: {self.heart_rate if hasattr(self, 'heart_rate') else 'N/A'} BPM
            - Condition: {self.condition if hasattr(self, 'condition') else 'N/A'}
            - AI Confidence: {self.confidence if hasattr(self, 'confidence') else 'N/A'}%
            
            This report includes:
            - Clinical interpretation
            - Doctor recommendations
            - WHO guidelines reference
            - ECG waveform visualization
            
            Please share this report with your healthcare provider for proper evaluation.
            
            This is an AI-generated analysis. Clinical decisions should be made by qualified healthcare professionals.
            
            Best regards,
            ECG AI Analysis System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach files
            for file_path in report_files:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename={os.path.basename(file_path)}'
                        )
                        msg.attach(part)
                        
            # Send email (you'll need to configure SMTP)
            # Uncomment and configure below:
            
            # server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            # server.starttls()
            # server.login("your_email@gmail.com", "your_app_password")
            # server.send_message(msg)
            # server.quit()
            
            print(f"✅ Email prepared for: {to_email}")
            print(f"   Attachments: {', '.join(report_files)}")
            print("\n⚠️  To actually send emails, configure SMTP settings:")
            print("   1. Use Gmail with App Password")
            print("   2. Update email_sender.py with your credentials")
            
            return True
            
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            return False
            
    def set_results(self, heart_rate, condition, confidence):
        """Store results for email body"""
        self.heart_rate = heart_rate
        self.condition = condition
        self.confidence = confidence
