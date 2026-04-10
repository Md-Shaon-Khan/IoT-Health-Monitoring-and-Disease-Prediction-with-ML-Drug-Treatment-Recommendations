# ============================================================
#  ECG Predictor — Ensemble (ResNet + Inception + Transformer)
#  Folder: ECG\predict_ecg.py
#
#  কীভাবে চালাবে:
#    python predict_ecg.py --input cleaned_output\cleaned_ecg_xxx.csv
#
#  অথবা raw CSV সরাসরি দিলে auto-clean করবে:
#    python predict_ecg.py --input raw_output\ecg_xxx.csv --raw
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import cleaning function
from ECG.clean_ecg import clean_ecg_data

def load_models():
    """Load all three ECG models"""
    try:
        # Model paths
        model_dir = "../ECG Saved Model Download from Kaggle"
        resnet_path = os.path.join(model_dir, "resnet_final.keras")
        inception_path = os.path.join(model_dir, "inception_final.keras")
        transformer_path = os.path.join(model_dir, "transformer_final.keras")

        print("Loading ECG models...")

        # Load models
        resnet_model = tf.keras.models.load_model(resnet_path)
        inception_model = tf.keras.models.load_model(inception_path)
        transformer_model = tf.keras.models.load_model(transformer_path)

        print("✅ All models loaded successfully!")
        return resnet_model, inception_model, transformer_model

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Make sure model files exist in 'ECG Saved Model Download from Kaggle' folder")
        return None, None, None

def preprocess_ecg_data(df, sequence_length=5000):
    """Preprocess ECG data for model input"""
    try:
        # Assuming ECG data is in a column named 'ecg_signal' or similar
        # Adjust column name based on your data
        ecg_columns = [col for col in df.columns if 'ecg' in col.lower() or 'signal' in col.lower()]

        if not ecg_columns:
            # If no ECG column found, use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in ECG data")
            ecg_col = numeric_cols[0]
        else:
            ecg_col = ecg_columns[0]

        print(f"Using ECG column: {ecg_col}")

        # Extract ECG signal
        ecg_signal = df[ecg_col].values

        # Normalize the signal
        scaler = StandardScaler()
        ecg_normalized = scaler.fit_transform(ecg_signal.reshape(-1, 1)).flatten()

        # Ensure minimum length
        if len(ecg_normalized) < sequence_length:
            # Pad with zeros if too short
            padding = np.zeros(sequence_length - len(ecg_normalized))
            ecg_normalized = np.concatenate([ecg_normalized, padding])
        elif len(ecg_normalized) > sequence_length:
            # Truncate if too long
            ecg_normalized = ecg_normalized[:sequence_length]

        # Reshape for model input (batch_size, sequence_length, 1)
        ecg_input = ecg_normalized.reshape(1, sequence_length, 1)

        return ecg_input, ecg_col

    except Exception as e:
        print(f"❌ Error preprocessing ECG data: {e}")
        return None, None

def predict_ecg_ensemble(ecg_input, resnet_model, inception_model, transformer_model):
    """Make prediction using ensemble of three models"""
    try:
        print("🔍 Making predictions with ensemble model...")

        # Get predictions from each model
        resnet_pred = resnet_model.predict(ecg_input, verbose=0)
        inception_pred = inception_model.predict(ecg_input, verbose=0)
        transformer_pred = transformer_model.predict(ecg_input, verbose=0)

        # Ensemble prediction (average)
        ensemble_pred = (resnet_pred + inception_pred + transformer_pred) / 3

        # Get the predicted class
        predicted_class = np.argmax(ensemble_pred, axis=1)[0]

        # Get confidence score
        confidence = np.max(ensemble_pred, axis=1)[0]

        # ECG classes (adjust based on your model's classes)
        ecg_classes = {
            0: "Normal",
            1: "Atrial Fibrillation",
            2: "Ventricular Tachycardia",
            3: "Myocardial Infarction",
            4: "Bundle Branch Block"
        }

        result = {
            'predicted_class': predicted_class,
            'class_name': ecg_classes.get(predicted_class, f"Class {predicted_class}"),
            'confidence': float(confidence),
            'individual_predictions': {
                'resnet': float(np.max(resnet_pred)),
                'inception': float(np.max(inception_pred)),
                'transformer': float(np.max(transformer_pred))
            }
        }

        return result

    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='ECG Prediction using Ensemble Model')
    parser.add_argument('--input', required=True, help='Path to ECG CSV file')
    parser.add_argument('--raw', action='store_true', help='Input is raw ECG data (will auto-clean)')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    print(f"📊 Processing ECG file: {args.input}")

    try:
        # Load ECG data
        df = pd.read_csv(args.input)
        print(f"✅ Loaded {len(df)} rows of ECG data")

        # Clean data if raw
        if args.raw:
            print("🧹 Cleaning raw ECG data...")
            df = clean_ecg_data(df)
            print("✅ Data cleaned")

        # Load models
        resnet_model, inception_model, transformer_model = load_models()
        if resnet_model is None:
            return

        # Preprocess data
        ecg_input, ecg_col = preprocess_ecg_data(df)
        if ecg_input is None:
            return

        # Make prediction
        result = predict_ecg_ensemble(ecg_input, resnet_model, inception_model, transformer_model)
        if result is None:
            return

        # Print results
        print("\n" + "="*50)
        print("🫀 ECG PREDICTION RESULTS")
        print("="*50)
        print(f"📁 Input File: {args.input}")
        print(f"📊 Data Column: {ecg_col}")
        print(f"🔍 Predicted Class: {result['class_name']}")
        print(".2f")
        print("\n📈 Individual Model Confidences:")
        print(".2f")
        print(".2f")
        print(".2f")
        print("="*50)

        # Save results to file
        result_file = args.input.replace('.csv', '_prediction.txt')
        with open(result_file, 'w') as f:
            f.write("ECG Prediction Results\n")
            f.write("="*50 + "\n")
            f.write(f"Input File: {args.input}\n")
            f.write(f"Predicted Class: {result['class_name']}\n")
            f.write(f"Confidence: {result['confidence']:.2f}\n")
            f.write(f"Individual Predictions: {result['individual_predictions']}\n")

        print(f"💾 Results saved to: {result_file}")

    except Exception as e:
        print(f"❌ Error processing ECG data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()# ============================================================
#  ECG Predictor — Ensemble (ResNet + Inception + Transformer)
#  Folder: ECG\predict_ecg.py
#
#  কীভাবে চালাবে:
#    python predict_ecg.py --input cleaned_output\cleaned_ecg_xxx.csv
#
#  অথবা raw CSV সরাসরি দিলে auto-clean করবে:
#    python predict_ecg.py --input raw_output\ecg_xxx.csv --raw
#
#  60s signal হলে → 6 complete 10s segment → 6 prediction → final summary
# ============================================================

import numpy as np
import pandas as pd
import os
import sys
import argparse
import json
import tensorflow as tf
from tensorflow.keras import layers
from scipy.signal import butter, filtfilt, iirnotch

# ============================================================
#  Register custom PositionalEncoding layer (must match training)
# ============================================================
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    """Positional encoding layer using pure TensorFlow ops (compatible with symbolic tensors)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]          # (seq_len, 1)
        dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]               # (1, d_model)
        angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
        angles = positions * angle_rates                                        # (seq_len, d_model)
        # Apply sin to even indices, cos to odd indices
        even_mask = tf.cast(tf.math.floormod(dims, 2) == 0, tf.float32)
        odd_mask = 1 - even_mask
        angles = tf.sin(angles) * even_mask + tf.cos(angles) * odd_mask
        return x + angles[tf.newaxis, :, :]                                     # broadcast over batch

    def get_config(self):
        return super().get_config()


# ── Config ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "ECG Saved Model Download from Kaggle")

WINDOW_SIZE   = 3600   # 10s × 360Hz
SAMPLING_RATE = 360

CLASS_NAMES = [
    "Normal",
    "Supraventricular",
    "Ventricular",
    "Conduction Disorder",
    "Myocardial Infarction",
    "Hypertrophy",
    "Ischemia/ST-T",
    "Atrial Fibrillation"
]

W_RESNET      = 0.45
W_INCEPTION   = 0.35
W_TRANSFORMER = 0.20


# ── Load models (lazy, once) ───────────────────────────────
_MODELS = None

def load_models():
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    model_files = {
        "resnet":      "resnet_final.keras",
        "inception":   "inception_final.keras",
        "transformer": "transformer_final.keras",
    }

    custom_objects = {'PositionalEncoding': PositionalEncoding}

    loaded = {}
    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"❌ Model not found: {path}")
            print(f"   Kaggle থেকে download করে 'ECG Saved Model Download from Kaggle' folder এ রাখো।")
            sys.exit(1)
        print(f"  Loading {name}...", end='', flush=True)
        loaded[name] = tf.keras.models.load_model(path, custom_objects=custom_objects)
        print(" ✅")

    _MODELS = loaded
    return _MODELS


# ── Preprocess a single 10s window ────────────────────────
def preprocess_window(signal, fs=SAMPLING_RATE):
    """Training এর সাথে identical preprocessing।"""
    nyq  = 0.5 * fs
    low  = 0.5 / nyq
    high = min(45.0 / nyq, 0.99)

    b, a = butter(3, [low, high], btype='bandpass')
    sig  = filtfilt(b, a, signal)

    w0 = 50.0 / nyq
    if w0 < 1.0:
        bn, an = iirnotch(w0, 30)
        sig = filtfilt(bn, an, sig)

    mean = np.mean(sig)
    std  = np.std(sig)
    sig  = ((sig - mean) / (std + 1e-8)).astype(np.float32)
    return sig


# ── Predict a single 10s window ───────────────────────────
def predict_segment(window, models):
    """একটা 10s segment এর জন্য ensemble prediction।"""
    x = window.reshape(1, WINDOW_SIZE, 1).astype(np.float32)

    p_r = models["resnet"].predict(x, verbose=0)[0]
    p_i = models["inception"].predict(x, verbose=0)[0]
    p_t = models["transformer"].predict(x, verbose=0)[0]

    ensemble = W_RESNET * p_r + W_INCEPTION * p_i + W_TRANSFORMER * p_t
    class_idx = int(np.argmax(ensemble))
    confidence = float(ensemble[class_idx])

    return {
        "class_idx":  class_idx,
        "prediction": CLASS_NAMES[class_idx],
        "confidence": confidence,
        "class_probs": {CLASS_NAMES[i]: float(ensemble[i]) for i in range(len(CLASS_NAMES))}
    }


# ── Load cleaned CSV ───────────────────────────────────────
def load_signal(filepath):
    """Cleaned CSV থেকে signal load করে।"""
    try:
        df = pd.read_csv(filepath, comment='#', header=None)
    except Exception as e:
        print(f"❌ Cannot read CSV: {e}")
        return None

    # single column বা multi-column
    if df.shape[1] == 1:
        raw = df.iloc[:, 0]
    else:
        # header row detect
        first_val = df.iloc[0, -1]
        try:
            float(str(first_val))
            raw = df.iloc[:, -1]
        except ValueError:
            raw = df.iloc[1:, -1]

    raw = pd.to_numeric(raw, errors='coerce').dropna()
    return raw.values.astype(np.float32)


# ── Main prediction pipeline ───────────────────────────────
def predict_full_signal(signal_path, raw_mode=False):
    """পুরো signal কে 10s segment এ ভাগ করে predict করে।"""

    # Raw mode হলে auto-clean
    if raw_mode:
        print("🔧 Auto-cleaning raw signal...")
        # Import inside to avoid circular dependency if needed
        from clean_ecg import clean_pipeline
        cleaned_path = clean_pipeline(signal_path)
        if cleaned_path is None:
            return None
        signal_path = cleaned_path

    signal = load_signal(signal_path)
    if signal is None:
        return None

    total_samples = len(signal)
    total_seconds = total_samples / SAMPLING_RATE
    n_segments    = int(total_samples // WINDOW_SIZE)

    print(f"\n📊 Signal info:")
    print(f"   Total samples : {total_samples}")
    print(f"   Duration      : {total_seconds:.1f}s")
    print(f"   Segments (10s): {n_segments}")

    if n_segments == 0:
        print(f"❌ Signal too short for even 1 segment. Need ≥ {WINDOW_SIZE} samples.")
        return None

    # Remainder (leftover < 10s) warning
    leftover = total_samples - (n_segments * WINDOW_SIZE)
    if leftover > 0:
        print(f"   ⚠️  Last {leftover/SAMPLING_RATE:.1f}s discarded (< 10s)")

    print("\n🤖 Loading ensemble models...")
    models = load_models()

    print(f"\n⚡ Predicting {n_segments} segment(s)...\n")
    print(f"{'Seg':<5} {'Start':>7} {'End':>7} {'Prediction':<22} {'Confidence':>11}  Status")
    print("─" * 65)

    results = []
    all_class_probs = np.zeros(len(CLASS_NAMES))

    for seg_idx in range(n_segments):
        start_sample = seg_idx * WINDOW_SIZE
        end_sample   = start_sample + WINDOW_SIZE
        start_sec    = start_sample / SAMPLING_RATE
        end_sec      = end_sample   / SAMPLING_RATE

        window = signal[start_sample:end_sample]
        window = preprocess_window(window)   # re-preprocess each segment

        result = predict_segment(window, models)
        result["seg"]     = seg_idx + 1
        result["start_t"] = round(start_sec, 1)
        result["end_t"]   = round(end_sec,   1)

        all_class_probs += np.array([result["class_probs"][c] for c in CLASS_NAMES])

        is_normal = result["prediction"] == "Normal"
        status    = "✅ Normal" if is_normal else "⚠️  ABNORMAL"

        print(f"#{seg_idx+1:<4} {start_sec:>6.1f}s  {end_sec:>6.1f}s  "
              f"{result['prediction']:<22} {result['confidence']*100:>9.1f}%  {status}")

        results.append(result)

    # ── Summary ──────────────────────────────────────────────
    avg_probs = all_class_probs / n_segments
    top_idx   = int(np.argmax(avg_probs))
    top_condition = CLASS_NAMES[top_idx]
    top_prob      = float(avg_probs[top_idx]) * 100

    normal_count   = sum(1 for r in results if r["prediction"] == "Normal")
    abnormal_count = n_segments - normal_count

    print("\n" + "═" * 65)
    print("  📋 FINAL SUMMARY")
    print("═" * 65)
    print(f"  Overall Condition : {top_condition}")
    print(f"  Avg Probability   : {top_prob:.1f}%")
    print(f"  Normal Segments   : {normal_count} / {n_segments}")
    print(f"  Abnormal Segments : {abnormal_count} / {n_segments}")
    print()
    print("  Class Probabilities (avg across all segments):")
    for i, cls in enumerate(CLASS_NAMES):
        bar = "█" * int(avg_probs[i] * 30)
        print(f"    {cls:<25} {avg_probs[i]*100:>5.1f}% {bar}")
    print("═" * 65)

    # ── Return structured result (for API use) ─────────
    return {
        "top_condition":  top_condition,
        "top_prob":       round(top_prob, 2),
        "normal_count":   normal_count,
        "abnormal_count": abnormal_count,
        "total_segments": n_segments,
        "class_probs":    {CLASS_NAMES[i]: round(float(avg_probs[i]) * 100, 2)
                           for i in range(len(CLASS_NAMES))},
        "segments": [
            {
                "seg":        r["seg"],
                "start_t":    r["start_t"],
                "end_t":      r["end_t"],
                "prediction": r["prediction"],
                "confidence": round(r["confidence"], 4),
            }
            for r in results
        ]
    }


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Ensemble Predictor")
    parser.add_argument("--input", type=str, required=True,
                        help="Cleaned (or raw with --raw) ECG CSV file")
    parser.add_argument("--raw",   action="store_true",
                        help="Input is raw — auto-clean before predicting")
    parser.add_argument("--json",  action="store_true",
                        help="Print result as JSON (for API use)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        sys.exit(1)

    result = predict_full_signal(args.input, raw_mode=args.raw)

    if result and args.json:
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(result, indent=2))