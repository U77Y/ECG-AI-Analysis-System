"""
AI Model for ECG Classification
Kutumia Deep Learning kutambua matatizo ya moyo
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class ECGAnalyzer:
    """AI Model for ECG Analysis"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, ecg_segment):
        """
        Extract features from ECG segment
        Features: statistical and morphological
        """
        features = []
        
        for segment in ecg_segment:
            # Statistical features
            mean_val = np.mean(segment)
            std_val = np.std(segment)
            max_val = np.max(segment)
            min_val = np.min(segment)
            rms_val = np.sqrt(np.mean(segment**2))
            
            # Combine features
            segment_features = [
                mean_val,
                std_val,
                max_val,
                min_val,
                rms_val,
                np.percentile(segment, 25),  # 25th percentile
                np.percentile(segment, 50),  # Median
                np.percentile(segment, 75)   # 75th percentile
            ]
            
            features.append(segment_features)
        
        return np.array(features)
    
    def train(self, X_train, y_train):
        """
        Train the model with ECG data
        X_train: ECG segments
        y_train: Labels (0=normal, 1=abnormal)
        """
        # Extract features
        features = self.extract_features(X_train)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled, y_train)
        self.is_trained = True
        
        print(f"✅ Model trained successfully!")
        print(f"   Training samples: {len(X_train)}")
        
    def predict(self, ecg_signal):
        """
        Predict if ECG signal is normal or abnormal
        Returns: prediction and confidence
        """
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        
        # Extract features
        features = self.extract_features([ecg_signal])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled)[0])
        
        return {
            'prediction': int(prediction),
            'condition': 'Normal' if prediction == 0 else 'Abnormal',
            'confidence': float(confidence)
        }
    
    def predict_heart_condition(self, ecg_signal):
        """
        Alias for predict method
        """
        return self.predict(ecg_signal)
