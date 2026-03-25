"""
Training Pipeline for ECG AI System
"""

import numpy as np
from src.preprocessing import ECGPreprocessor
from src.model import ECGAnalyzer

def generate_sample_data():
    """Generate sample ECG data for testing"""
    print("📊 Generating sample ECG data...")
    
    t = np.linspace(0, 10, 3600)
    normal_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.random.randn(len(t))
    
    abnormal_ecg = np.sin(2 * np.pi * 1.8 * t) + 0.8 * np.random.randn(len(t))
    abnormal_ecg[1000:1200] += 2.0
    
    X_train = []
    y_train = []
    
    for i in range(50):
        X_train.append(normal_ecg + 0.1 * np.random.randn(len(normal_ecg)))
        y_train.append(0)
    
    for i in range(50):
        X_train.append(abnormal_ecg + 0.1 * np.random.randn(len(abnormal_ecg)))
        y_train.append(1)
    
    return np.array(X_train), np.array(y_train)

def main():
    print("=" * 50)
    print("ECG AI Analysis System - Training")
    print("=" * 50)
    
    X_train, y_train = generate_sample_data()
    print(f"✅ Generated {len(X_train)} training samples")
    
    preprocessor = ECGPreprocessor(sampling_rate=360)
    
    X_filtered = []
    for signal in X_train:
        filtered = preprocessor.remove_noise(signal)
        X_filtered.append(filtered)
    
    X_filtered = np.array(X_filtered)
    print("✅ Data preprocessing completed")
    
    model = ECGAnalyzer()
    model.train(X_filtered, y_train)
    
    print("\n📊 Testing with sample ECG...")
    test_ecg = X_filtered[0]
    result = model.predict(test_ecg)
    
    print(f"   Prediction: {result['condition']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    
    print("\n🎉 Training completed successfully!")
    
    return model

if __name__ == "__main__":
    model = main()
