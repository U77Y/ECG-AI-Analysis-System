"""
Clinical Decision Support Module
Ushauri kwa Doctor na Mgonjwa
"""

import numpy as np
from datetime import datetime

class ClinicalAdvisor:
    """Class ya kutoa ushauri wa kitabibu na maelezo kwa mgonjwa"""
    
    def __init__(self):
        self.who_references = self._get_who_references()
        self.clinical_guidelines = self._get_clinical_guidelines()
        
    def _get_who_references(self):
        """WHO references za kina"""
        return {
            'heart_rate': {
                'normal': "WHO defines normal resting heart rate as 60-100 beats per minute (bpm) for adults.",
                'bradycardia': "Bradycardia is defined as heart rate < 60 bpm. WHO notes that while common in athletes, it may indicate underlying pathology.",
                'tachycardia': "Tachycardia is defined as heart rate > 100 bpm. WHO guidelines recommend evaluation for causes including arrhythmias, anxiety, or cardiac conditions."
            },
            'ecg_interpretation': {
                'normal': "According to WHO, a normal ECG shows regular rhythm with normal P wave, QRS complex (80-100ms), and T wave morphology.",
                'abnormal': "WHO indicates that ECG abnormalities require clinical correlation with patient history and physical examination.",
                'rhythm': "WHO emphasizes that rhythm interpretation should consider the 12-lead ECG for comprehensive assessment."
            },
            'risk_factors': {
                'hypertension': "WHO identifies hypertension as a major risk factor for cardiovascular disease.",
                'diabetes': "WHO notes diabetes significantly increases cardiovascular risk.",
                'smoking': "WHO states tobacco use is a leading cause of cardiovascular disease."
            },
            'prevention': {
                'lifestyle': "WHO recommends regular physical activity, healthy diet, and smoking cessation for cardiovascular prevention.",
                'monitoring': "WHO guidelines suggest regular ECG monitoring for patients with known cardiovascular risk factors."
            }
        }
        
    def _get_clinical_guidelines(self):
        """Clinical guidelines for doctors"""
        return {
            'action_needed': [
                "Correlate ECG findings with patient symptoms",
                "Consider additional testing if indicated",
                "Review patient history and risk factors",
                "Document findings in medical records"
            ],
            'referral_criteria': [
                "Persistent abnormalities requiring cardiology consultation",
                "Symptoms suggestive of ischemia",
                "Complex arrhythmias requiring specialist management"
            ],
            'follow_up': [
                "Repeat ECG if clinically indicated",
                "Monitor for symptom progression",
                "Assess treatment response"
            ]
        }
        
    def get_doctor_recommendations(self, result, heart_rate, patient_info=None):
        """
        Ushauri wa kina kwa Doctor
        """
        recommendations = {
            'summary': "",
            'clinical_interpretation': "",
            'recommended_actions': [],
            'referral_needed': False,
            'urgent_care_needed': False,
            'medication_considerations': [],
            'follow_up_timeline': ""
        }
        
        # Summary based on findings
        if result['condition'] == 'Normal' and 60 <= heart_rate <= 100:
            recommendations['summary'] = "Normal ECG with regular sinus rhythm. No acute abnormalities detected."
            recommendations['clinical_interpretation'] = "This ECG shows normal morphology, rate, and rhythm. Patient is within normal parameters."
            recommendations['follow_up_timeline'] = "Routine follow-up as clinically indicated (6-12 months)"
        elif heart_rate < 60:
            recommendations['summary'] = f"Sinus bradycardia detected with heart rate of {heart_rate:.1f} bpm."
            recommendations['clinical_interpretation'] = f"WHO Reference: {self.who_references['heart_rate']['bradycardia']}"
            recommendations['recommended_actions'].append("Evaluate for symptoms: dizziness, syncope, fatigue")
            recommendations['recommended_actions'].append("Consider medication review (beta-blockers, calcium channel blockers)")
            recommendations['medication_considerations'].append("Review current medications that may affect heart rate")
            if heart_rate < 50:
                recommendations['referral_needed'] = True
                recommendations['urgent_care_needed'] = heart_rate < 45
        elif heart_rate > 100:
            recommendations['summary'] = f"Sinus tachycardia detected with heart rate of {heart_rate:.1f} bpm."
            recommendations['clinical_interpretation'] = f"WHO Reference: {self.who_references['heart_rate']['tachycardia']}"
            recommendations['recommended_actions'].append("Assess for underlying causes: fever, dehydration, anxiety, pain")
            recommendations['recommended_actions'].append("Consider thyroid function testing if persistent")
            if heart_rate > 120:
                recommendations['referral_needed'] = True
                
        if result['condition'] == 'Abnormal':
            recommendations['summary'] += " Abnormal ECG pattern detected."
            recommendations['recommended_actions'].append("Further evaluation with cardiology consultation recommended")
            recommendations['recommended_actions'].append("Consider additional diagnostic testing (echocardiogram, Holter monitor)")
            recommendations['referral_needed'] = True
            
        recommendations['recommended_actions'].extend(self.clinical_guidelines['action_needed'][:3])
        
        return recommendations
        
    def get_patient_explanation(self, result, heart_rate):
        """
        Maelezo rahisi kwa mgonjwa (patient-friendly language)
        """
        explanation = {
            'what_it_means': "",
            'condition_explained': "",
            'lifestyle_advice': [],
            'when_to_seek_help': [],
            'confidence_message': ""
        }
        
        # Simple explanation
        if result['condition'] == 'Normal' and 60 <= heart_rate <= 100:
            explanation['what_it_means'] = "Your heart is beating normally. The electrical activity of your heart shows a regular pattern."
            explanation['condition_explained'] = "✅ Normal sinus rhythm - this is what we expect to see in a healthy heart."
            explanation['confidence_message'] = f"We are {result['confidence']:.1%} confident in this analysis."
        elif heart_rate < 60:
            explanation['what_it_means'] = "Your heart is beating slower than average."
            explanation['condition_explained'] = "⚡ Slow Heart Rate (Bradycardia): This can be normal for athletes, but if you feel dizzy or tired, please consult your doctor."
            explanation['lifestyle_advice'].append("Stay hydrated")
            explanation['lifestyle_advice'].append("Avoid excessive caffeine")
            explanation['when_to_seek_help'].append("If you feel dizzy, faint, or very tired")
        elif heart_rate > 100:
            explanation['what_it_means'] = "Your heart is beating faster than average."
            explanation['condition_explained'] = "⚡ Fast Heart Rate (Tachycardia): This can happen with stress, fever, or exercise. If persistent, please see your doctor."
            explanation['lifestyle_advice'].append("Practice deep breathing and relaxation")
            explanation['lifestyle_advice'].append("Reduce caffeine and alcohol")
            explanation['when_to_seek_help'].append("If you feel palpitations, chest pain, or shortness of breath")
            
        if result['condition'] == 'Abnormal':
            explanation['condition_explained'] += " We detected an unusual pattern that needs medical review."
            explanation['when_to_seek_help'].append("Please share this report with your doctor for proper evaluation")
            
        explanation['lifestyle_advice'].append("Maintain a healthy diet and regular exercise")
        explanation['lifestyle_advice'].append("Follow WHO guidelines: 150 minutes of moderate activity per week")
        
        return explanation
        
    def get_full_who_references(self):
        """
        Full WHO references for report end
        """
        references = """
        ============================================================
        WORLD HEALTH ORGANIZATION (WHO) REFERENCES
        ============================================================
        
        1. WHO Guidelines for Cardiovascular Disease Prevention:
           - Regular physical activity (150 minutes/week moderate intensity)
           - Healthy diet rich in fruits, vegetables, and whole grains
           - Limit salt intake (<5g/day)
           - Avoid tobacco use
           - Limit alcohol consumption
        
        2. WHO ECG Interpretation Standards:
           - Normal heart rate: 60-100 bpm
           - PR interval: 120-200 ms
           - QRS duration: <120 ms
           - QT interval: <460 ms (corrected)
        
        3. WHO Cardiovascular Risk Factors:
           - Hypertension (BP >140/90 mmHg)
           - Diabetes mellitus
           - Hyperlipidemia
           - Obesity (BMI >30)
           - Physical inactivity
           - Tobacco use
        
        4. WHO Recommendations for ECG Monitoring:
           - Regular screening for high-risk individuals
           - 12-lead ECG as standard for comprehensive assessment
           - Clinical correlation required for accurate diagnosis
        
        5. WHO Sustainable Development Goals (SDG 3.4):
           - Reduce premature mortality from cardiovascular diseases by 1/3 by 2030
           - Strengthen prevention and treatment of heart conditions
        
        For more information: https://www.who.int/health-topics/cardiovascular-diseases
        
        ============================================================
        """
        return references
