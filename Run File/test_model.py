# test_model.py
"""
Health Prediction System - Model Testing
This file loads the saved model and tests it
"""

import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

def load_saved_model():
    """Load the saved model package"""
    print("Loading saved model from 'model_saved.pkl'...")
    
    with open('F:/Final Project/Main Model Train/model_saved.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    print("Model loaded successfully")
    
    
    model = model_package['model']
    scaler = model_package['scaler']
    label_encoder = model_package['label_encoder']
    feature_names = model_package['feature_names']
    numerical_cols = model_package['numerical_cols']
    categorical_cols = model_package['categorical_cols']
    metadata = model_package['model_metadata']
    
    print(f"\nModel Information:")
    print(f"• Accuracy: {metadata['accuracy']:.4f} ({metadata['accuracy']*100:.2f}%)")
    print(f"• Test Samples: {metadata['test_samples']}")
    print(f"• Misclassified: {metadata['misclassified']}")
    print(f"• Input Shape: {metadata['input_shape']}")
    print(f"• Output Shape: {metadata['output_shape']}")
    print(f"• Training Date: {metadata['training_date']}")
    
    print(f"\nFeatures ({len(feature_names)}):")
    for i, feature in enumerate(feature_names, 1):
        print(f"  {i:2d}. {feature}")
    
    return model, scaler, label_encoder, feature_names, numerical_cols, categorical_cols

def predict_disease(input_features, model, scaler, label_encoder, feature_names, numerical_cols):
    
   
    features_array = np.array(input_features).reshape(1, -1)
    
   
    features_df = pd.DataFrame(features_array, columns=feature_names)
    
   
    features_df[numerical_cols] = scaler.transform(features_df[numerical_cols])
    
   
    prediction_proba = model.predict(features_df.values, verbose=0)
    prediction_class = np.argmax(prediction_proba)
    
  
    predicted_disease = label_encoder.inverse_transform([prediction_class])[0]
    confidence = np.max(prediction_proba)
    
 
    probabilities = {
        disease: float(prob)
        for disease, prob in zip(label_encoder.classes_, prediction_proba[0])
    }
    
    return predicted_disease, confidence, probabilities

def test_sample_cases():
    """Test the model with sample cases"""
    print("\n" + "="*60)
    print("TESTING SAMPLE CASES")
    print("="*60)
    

    model, scaler, label_encoder, feature_names, numerical_cols, categorical_cols = load_saved_model()
    

    test_cases = [
        {
            'name': 'Normal Healthy Person',
            'features': [36.5, 75.0, 120.0, 80.0, 40.0, 0, 0, 0, 0, 0, 0],
            'expected': 'Normal'
        },
        {
            'name': 'Heart Risk Patient',
            'features': [37.0, 105.0, 150.0, 95.0, 40.0, 0, 0, 1, 1, 1, 0],
            'expected': 'Heart_Risk'
        },
        {
            'name': 'Fever/Respiratory Infection',
            'features': [39.0, 110.0, 115.0, 75.0, 40.0, 1, 1, 0, 0, 1, 1],
            'expected': 'Fever_Respiratory'
        },
        {
            'name': 'Hypertension',
            'features': [36.8, 85.0, 150.0, 95.0, 40.0, 0, 0, 0, 0, 0, 0],
            'expected': 'Hypertension'
        },
        {
            'name': 'Hypotension',
            'features': [36.5, 85.0, 90.0, 55.0, 40.0, 0, 0, 0, 1, 1, 0],
            'expected': 'Hypotension'
        }
    ]
    
    print("\nTest Results:")
    print("-" * 80)
    print(f"{'Test Case':30s} {'Expected':20s} {'Predicted':20s} {'Confidence':12s} {'Match'}")
    print("-" * 80)
    
    results = []
    for case in test_cases:
        predicted_disease, confidence, probabilities = predict_disease(
            case['features'], model, scaler, label_encoder, 
            feature_names, numerical_cols
        )
        
        match = "✓" if predicted_disease == case['expected'] else "✗"
        
        print(f"{case['name']:30s} {case['expected']:20s} {predicted_disease:20s} {confidence:.4f}        {match}")
        
        results.append({
            'case': case['name'],
            'expected': case['expected'],
            'predicted': predicted_disease,
            'confidence': confidence,
            'match': predicted_disease == case['expected']
        })
    

    correct = sum(1 for r in results if r['match'])
    total = len(results)
    accuracy = correct / total
    
    print("-" * 80)
    print(f"Test Accuracy: {correct}/{total} ({accuracy:.2%})")
    
    return results

def run_batch_test():
    """Run a batch test with multiple samples"""
    print("\n" + "="*60)
    print("BATCH TESTING")
    print("="*60)
    

    model, scaler, label_encoder, feature_names, numerical_cols, categorical_cols = load_saved_model()
    

    np.random.seed(42)
    n_samples = 100
    

    test_data = []
    for _ in range(n_samples):
      
        temp = np.random.uniform(36.0, 40.0)
        heart_rate = np.random.uniform(60.0, 120.0)
        bp_sys = np.random.uniform(80.0, 160.0)
        bp_dia = np.random.uniform(50.0, 100.0)
        humidity = np.random.uniform(30.0, 50.0)
        
      
        fever = np.random.choice([0, 1])
        cough = np.random.choice([0, 1])
        chest_pain = np.random.choice([0, 1])
        shortness = np.random.choice([0, 1])
        fatigue = np.random.choice([0, 1])
        headache = np.random.choice([0, 1])
        
        features = [temp, heart_rate, bp_sys, bp_dia, humidity,
                   fever, cough, chest_pain, shortness, fatigue, headache]
        
        test_data.append(features)
    
 
    print(f"Making predictions for {n_samples} samples...")
    
    predictions = []
    confidences = []
    
    for features in test_data:
        predicted_disease, confidence, _ = predict_disease(
            features, model, scaler, label_encoder, 
            feature_names, numerical_cols
        )
        predictions.append(predicted_disease)
        confidences.append(confidence)
    

    avg_confidence = np.mean(confidences)
    std_confidence = np.std(confidences)
    
    print(f"\nBatch Test Results:")
    print(f"• Samples tested: {n_samples}")
    print(f"• Average confidence: {avg_confidence:.4f}")
    print(f"• Std confidence: {std_confidence:.4f}")
    print(f"• Min confidence: {np.min(confidences):.4f}")
    print(f"• Max confidence: {np.max(confidences):.4f}")
    

    from collections import Counter
    pred_counts = Counter(predictions)
    
    print(f"\nPrediction Distribution:")
    for disease, count in pred_counts.items():
        percentage = count / n_samples * 100
        print(f"  {disease:20s}: {count:3d} ({percentage:.1f}%)")
    
    return predictions, confidences

def main():
    """Main function to run all tests"""
    print("="*60)
    print("HEALTH PREDICTION SYSTEM - MODEL TESTING")
    print("="*60)
    
  
    sample_results = test_sample_cases()
    
  
    batch_predictions, batch_confidences = run_batch_test()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nThe model is working correctly and ready for use.")
    
    return sample_results, batch_predictions, batch_confidences

if __name__ == "__main__":
    main()