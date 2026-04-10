# ============================================================
#  ECG Data Cleaner
#  Folder: ECG\clean_ecg.py
#
#  কাজ: Raw ECG data কে clean করে model-ready format এ convert করে
# ============================================================

import pandas as pd
import numpy as np
from scipy import signal
import argparse
import os

def clean_ecg_data(df):
    """
    Clean raw ECG data by removing noise, artifacts, and normalizing

    Parameters:
    df (DataFrame): Raw ECG data

    Returns:
    DataFrame: Cleaned ECG data
    """
    try:
        print("🧹 Starting ECG data cleaning...")

        # Find ECG signal column
        ecg_columns = [col for col in df.columns if 'ecg' in col.lower() or 'signal' in col.lower()]

        if not ecg_columns:
            # Use first numeric column if no ECG column found
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in ECG data")
            ecg_col = numeric_cols[0]
        else:
            ecg_col = ecg_columns[0]

        print(f"📊 Processing ECG column: {ecg_col}")

        # Extract ECG signal
        ecg_signal = df[ecg_col].values

        # Step 1: Remove NaN and infinite values
        ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 2: Baseline wander removal using high-pass filter
        # Design high-pass filter to remove baseline wander (< 0.5 Hz)
        nyquist = 250  # Assuming 500 Hz sampling rate
        cutoff = 0.5
        b, a = signal.butter(4, cutoff/(nyquist), btype='high')
        ecg_filtered = signal.filtfilt(b, a, ecg_signal)

        # Step 3: Power line interference removal (50/60 Hz notch filter)
        # Design notch filter for 50 Hz
        notch_freq = 50.0
        quality_factor = 30.0
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, nyquist*2)
        ecg_filtered = signal.filtfilt(b_notch, a_notch, ecg_filtered)

        # Also filter 60 Hz for good measure
        notch_freq = 60.0
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, nyquist*2)
        ecg_filtered = signal.filtfilt(b_notch, a_notch, ecg_filtered)

        # Step 4: Muscle artifact removal using low-pass filter
        # Design low-pass filter (< 40 Hz for ECG)
        cutoff_lp = 40.0
        b_lp, a_lp = signal.butter(4, cutoff_lp/nyquist, btype='low')
        ecg_filtered = signal.filtfilt(b_lp, a_lp, ecg_filtered)

        # Step 5: Normalize the signal (Z-score normalization)
        ecg_mean = np.mean(ecg_filtered)
        ecg_std = np.std(ecg_filtered)

        if ecg_std > 0:
            ecg_normalized = (ecg_filtered - ecg_mean) / ecg_std
        else:
            ecg_normalized = ecg_filtered - ecg_mean

        # Step 6: Remove outliers using median absolute deviation
        median = np.median(ecg_normalized)
        mad = np.median(np.abs(ecg_normalized - median))
        threshold = 3.0  # 3 MAD threshold

        if mad > 0:
            outliers = np.abs(ecg_normalized - median) > (threshold * mad)
            ecg_normalized[outliers] = median  # Replace outliers with median

        # Create cleaned dataframe
        df_cleaned = df.copy()
        df_cleaned[ecg_col] = ecg_normalized

        print("✅ ECG data cleaning completed!")
        print(f"   - Original samples: {len(ecg_signal)}")
        print(f"   - Outliers removed: {np.sum(outliers) if 'outliers' in locals() else 0}")

        return df_cleaned

    except Exception as e:
        print(f"❌ Error cleaning ECG data: {e}")
        return df  # Return original data if cleaning fails

def main():
    parser = argparse.ArgumentParser(description='Clean ECG Data')
    parser.add_argument('--input', required=True, help='Path to raw ECG CSV file')
    parser.add_argument('--output', help='Path to save cleaned ECG data (optional)')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    print(f"📊 Loading raw ECG data from: {args.input}")

    try:
        # Load raw ECG data
        df_raw = pd.read_csv(args.input)
        print(f"✅ Loaded {len(df_raw)} rows of raw ECG data")

        # Clean the data
        df_cleaned = clean_ecg_data(df_raw)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Generate output path in cleaned_output folder
            filename = os.path.basename(args.input)
            clean_filename = f"cleaned_{filename}"
            output_path = os.path.join("cleaned_output", clean_filename)

            # Create cleaned_output directory if it doesn't exist
            os.makedirs("cleaned_output", exist_ok=True)

        # Save cleaned data
        df_cleaned.to_csv(output_path, index=False)
        print(f"💾 Cleaned ECG data saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error processing ECG data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()# ============================================================
#  ECG Signal Cleaner — Ultra Advanced Noise Cancellation
#  Folder: ECG\clean_ecg.py
#
#  Simplified usage:
#    python clean_ecg.py
#  (automatically finds the latest raw CSV in raw_output folder)
#
#  Advanced features:
#    - Wavelet denoising (multi-level DWT thresholding)
#    - Hampel filter for outlier removal (robust)
#    - Adaptive notch filter for 50 Hz and harmonics
#    - Baseline wander removal via cubic spline fitting
#    - Median filter for impulse spikes
#    - MAD-based outlier detection
#    - Flat-line interpolation
#    - Bandpass filtering (0.5-40 Hz)
#    - Automatic resampling to 360 Hz
# ============================================================

import numpy as np
import pandas as pd
import os
import argparse
import glob
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, medfilt
from scipy.interpolate import CubicSpline
from scipy.stats import median_abs_deviation
from math import gcd
from datetime import datetime

# Optional wavelet import
try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    print("⚠️ PyWavelets not installed. Wavelet denoising disabled. Install with: pip install PyWavelets")

# ── Config ─────────────────────────────────────────────────
TARGET_FS     = 360
WINDOW_SIZE   = 3600
MIN_SAMPLES   = WINDOW_SIZE

ADC_MIN       = 0
ADC_MAX       = 1023
LEAD_OFF_VAL  = 512
LEAD_OFF_TOLERANCE = 5
FLAT_WINDOW   = 72
HAMPEL_WINDOW = 11          # Window size for Hampel filter
HAMPEL_THRESH = 3.5         # Number of MADs for outlier detection
MAD_THRESH    = 5.0

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_output")
RAW_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_output")

# ── Helper: find latest raw file ──────────────────────────
def find_latest_raw_file():
    if not os.path.exists(RAW_DIR):
        return None
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not csv_files:
        return None
    latest = max(csv_files, key=os.path.getctime)
    return latest

# ── Load CSV ──────────────────────────────────────────────
def load_csv(filepath):
    print(f"\n📂 Loading: {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath, comment='#', header=None)
    except Exception as e:
        print(f"  ❌ CSV read failed: {e}")
        return None

    if df.shape[1] == 1:
        raw = df.iloc[:, 0]
    elif df.shape[1] >= 2:
        raw = df.iloc[:, -1]
    else:
        print("  ❌ Unrecognized CSV format.")
        return None

    raw = pd.to_numeric(raw, errors='coerce').dropna()
    if len(raw) == 0:
        print("  ❌ No numeric data found.")
        return None

    signal = raw.values.astype(np.float64)
    print(f"  ✅ Loaded {len(signal)} samples ({len(signal)/TARGET_FS:.1f}s at {TARGET_FS}Hz)")
    return signal

# ── Ultra‑advanced: Wavelet Denoising ─────────────────────
def wavelet_denoise(signal, wavelet='db4', level=4, method='soft'):
    """Multi-level wavelet denoising using universal threshold."""
    if not WAVELET_AVAILABLE:
        return signal
    # Decompose
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Estimate noise standard deviation from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if sigma < 1e-8:
        return signal
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    # Apply threshold to detail coefficients
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        if method == 'soft':
            new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        else:
            new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
    # Reconstruct
    denoised = pywt.waverec(new_coeffs, wavelet)
    # Trim to original length
    if len(denoised) > len(signal):
        denoised = denoised[:len(signal)]
    print(f"  ✅ Wavelet denoising (wavelet={wavelet}, level={level}, method={method})")
    return denoised

# ── Hampel filter for outlier removal ─────────────────────
def hampel_filter(signal, window_size=HAMPEL_WINDOW, n_sigmas=HAMPEL_THRESH):
    """Hampel identifier: replace outliers with median of window."""
    signal = signal.copy()
    n = len(signal)
    half_window = window_size // 2
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = signal[start:end]
        median = np.median(window)
        mad = median_abs_deviation(window)
        if mad == 0:
            continue
        if abs(signal[i] - median) > n_sigmas * mad:
            signal[i] = median
    print(f"  ✅ Hampel filter applied (window={window_size}, sigma={n_sigmas})")
    return signal

# ── Remove lead-off & saturation ──────────────────────────
def remove_lead_off_artifacts(signal):
    signal = signal.copy()
    n = len(signal)
    lead_off_mask = np.abs(signal - LEAD_OFF_VAL) <= LEAD_OFF_TOLERANCE
    sat_mask = (signal <= ADC_MIN + 2) | (signal >= ADC_MAX - 2)
    bad_mask = lead_off_mask | sat_mask
    bad_count = np.sum(bad_mask)
    if bad_count == 0:
        return signal
    indices = np.arange(n)
    good_idx = indices[~bad_mask]
    good_val = signal[~bad_mask]
    if len(good_val) >= 2:
        signal[bad_mask] = np.interp(indices[bad_mask], good_idx, good_val)
        print(f"  ✅ Fixed {bad_count} lead-off/saturation artifacts ({bad_count/n*100:.1f}%)")
    return signal

# ── Baseline wander removal (cubic spline) ────────────────
def remove_baseline_wander(signal, fs=TARGET_FS, cutoff=0.5):
    """Estimate baseline by low-pass filtering (or cubic spline on minima)."""
    # Use high-pass filter (already in bandpass) but also direct spline method
    # Find minima points (every 0.5 sec) and fit cubic spline
    signal = signal.copy()
    n = len(signal)
    step = int(fs * 0.5)  # every 0.5 second
    indices = np.arange(0, n, step)
    if len(indices) < 4:
        return signal
    min_vals = [np.min(signal[max(0, i-step//2):min(n, i+step//2)]) for i in indices]
    spline = CubicSpline(indices, min_vals, bc_type='natural')
    baseline = spline(np.arange(n))
    corrected = signal - baseline
    print(f"  ✅ Baseline wander removed (cubic spline, step={step} samples)")
    return corrected

# ── Adaptive notch filter for 50 Hz and harmonics ─────────
def adaptive_notch_filter(signal, fs=TARGET_FS, freqs=[50, 100, 150, 200], quality=30):
    """Remove multiple powerline harmonics."""
    filtered = signal.copy()
    nyq = 0.5 * fs
    for freq in freqs:
        w0 = freq / nyq
        if w0 < 1.0:
            b, a = iirnotch(w0, quality)
            filtered = filtfilt(b, a, filtered)
    print(f"  ✅ Adaptive notch filter applied (freqs: {freqs} Hz)")
    return filtered

# ── Median filter for spikes ──────────────────────────────
def median_filter_denoise(signal, kernel_size=5):
    filtered = medfilt(signal, kernel_size)
    print(f"  ✅ Median filter (kernel={kernel_size})")
    return filtered

# ── Flat-line detection ───────────────────────────────────
def fix_flatline_segments(signal):
    signal = signal.copy()
    n = len(signal)
    bad_mask = np.zeros(n, dtype=bool)
    i = 0
    flat_count = 0
    while i < n - FLAT_WINDOW:
        window = signal[i:i+FLAT_WINDOW]
        if np.std(window) < 0.5:
            bad_mask[i:i+FLAT_WINDOW] = True
            flat_count += 1
            i += FLAT_WINDOW
        else:
            i += 1
    count = np.sum(bad_mask)
    if count == 0:
        return signal
    indices = np.arange(n)
    good_idx = indices[~bad_mask]
    good_val = signal[~bad_mask]
    if len(good_val) >= 2:
        signal[bad_mask] = np.interp(indices[bad_mask], good_idx, good_val)
        print(f"  ✅ Fixed {count} flat-line samples in {flat_count} segments")
    return signal

# ── Bandpass filter (0.5 - 40 Hz) ─────────────────────────
def bandpass_filter(signal, fs=TARGET_FS, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    print(f"  ✅ Bandpass filter ({lowcut}-{highcut} Hz, order {order})")
    return filtered

# ── Resample ──────────────────────────────────────────────
def resample_signal(signal, from_fs, to_fs):
    if from_fs == to_fs:
        return signal
    g = gcd(int(from_fs), int(to_fs))
    up = int(to_fs) // g
    down = int(from_fs) // g
    resampled = resample_poly(signal, up, down)
    print(f"  ✅ Resampled {from_fs}Hz → {to_fs}Hz ({len(signal)} → {len(resampled)} samples)")
    return resampled

# ── Normalize ─────────────────────────────────────────────
def normalize_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        return signal
    normalized = ((signal - mean) / (std + 1e-8)).astype(np.float32)
    print(f"  ✅ Normalized (mean={mean:.2f}, std={std:.2f})")
    return normalized

# ── Save cleaned CSV ──────────────────────────────────────
def save_cleaned(signal, original_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(original_path))[0]
    out_path = os.path.join(output_dir, f"cleaned_{basename}.csv")
    pd.DataFrame(signal, columns=["ecg_value"]).to_csv(out_path, index=False)
    print(f"\n💾 Saved: {out_path}")
    print(f"   Total samples: {len(signal)}")
    print(f"   Duration     : {len(signal)/TARGET_FS:.1f}s")
    segments = len(signal) // WINDOW_SIZE
    print(f"   10s segments : {segments}")
    return out_path

# ── Main pipeline (ultra advanced) ────────────────────────
def clean_pipeline(input_path, input_fs=TARGET_FS):
    print("=" * 60)
    print("  ECG Signal Cleaning — Ultra Advanced Pipeline")
    print("=" * 60)

    signal = load_csv(input_path)
    if signal is None:
        return None

    print("\n🔧 Cleaning steps (ultra advanced):")

    # 1. Remove lead-off/saturation
    signal = remove_lead_off_artifacts(signal)

    # 2. Median filter for spikes
    signal = median_filter_denoise(signal, kernel_size=5)

    # 3. Hampel filter for outliers
    signal = hampel_filter(signal, window_size=11, n_sigmas=3.5)

    # 4. Flat-line correction
    signal = fix_flatline_segments(signal)

    # 5. Wavelet denoising (if available)
    if WAVELET_AVAILABLE:
        signal = wavelet_denoise(signal, wavelet='db4', level=4, method='soft')
    else:
        print("  ⚠️ Wavelet denoising skipped (install PyWavelets)")

    # 6. Baseline wander removal
    signal = remove_baseline_wander(signal, fs=TARGET_FS)

    # 7. Resample if needed
    if input_fs != TARGET_FS:
        signal = resample_signal(signal, input_fs, TARGET_FS)

    # 8. Bandpass filter (0.5-40 Hz)
    signal = bandpass_filter(signal, fs=TARGET_FS, lowcut=0.5, highcut=40.0, order=4)

    # 9. Adaptive notch filter (50, 100, 150, 200 Hz)
    signal = adaptive_notch_filter(signal, fs=TARGET_FS, freqs=[50, 100, 150, 200], quality=30)

    # 10. Normalize
    signal = normalize_signal(signal)

    if len(signal) < MIN_SAMPLES:
        print(f"\n❌ After cleaning, signal too short ({len(signal)} samples).")
        return None

    out_path = save_cleaned(signal, input_path, OUTPUT_DIR)
    print(f"\n✅ Ultra advanced cleaning complete!\n📌 Next step: python predict_ecg.py --input \"{out_path}\"")
    return out_path

# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Signal Cleaner (Ultra Advanced)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV file or folder path. If not given, uses latest file in raw_output/")
    parser.add_argument("--fs", type=int, default=TARGET_FS,
                        help=f"Input sampling rate in Hz (default: {TARGET_FS})")
    args = parser.parse_args()

    if args.input:
        input_path = args.input
    else:
        input_path = find_latest_raw_file()
        if input_path is None:
            print("❌ No raw CSV file found in 'raw_output' folder. Please record data first or specify --input.")
            exit(1)
        print(f"🔍 Auto-selected latest raw file: {os.path.basename(input_path)}")

    if os.path.isdir(input_path):
        csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"❌ No CSV files found in {input_path}")
            exit(1)
        print(f"Found {len(csv_files)} CSV file(s).")
        for csv_file in csv_files:
            full_path = os.path.join(input_path, csv_file)
            clean_pipeline(full_path, input_fs=args.fs)
    elif os.path.isfile(input_path):
        clean_pipeline(input_path, input_fs=args.fs)
    else:
        print(f"❌ Path not found: {input_path}")
        exit(1)