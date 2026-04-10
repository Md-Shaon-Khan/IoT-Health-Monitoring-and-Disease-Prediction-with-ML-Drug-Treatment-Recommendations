# ============================================================
#  ECG Data Collector
#  Folder: ECG\collect_ecg.py
#
#  কাজ: IoT devices থেকে ECG data collect করে CSV file এ save করে
# ============================================================

import serial
import pandas as pd
import time
import argparse
import os
from datetime import datetime
import threading
import queue

class ECGCollector:
    def __init__(self, port='COM3', baudrate=9600, duration=30):
        """
        Initialize ECG data collector

        Parameters:
        port (str): Serial port for Arduino/ESP32
        baudrate (int): Serial communication baud rate
        duration (int): Collection duration in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.duration = duration
        self.data_queue = queue.Queue()
        self.is_collecting = False
        self.serial_connection = None

    def connect_device(self):
        """Connect to ECG device via serial"""
        try:
            print(f"🔌 Connecting to ECG device on {self.port}...")
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Wait for connection to establish
            print("✅ Connected to ECG device!")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to ECG device: {e}")
            print("💡 Make sure:")
            print("   - Device is connected to the correct COM port")
            print("   - Baud rate matches device settings")
            print("   - No other program is using the port")
            return False

    def collect_data_thread(self):
        """Thread function to collect ECG data"""
        print("📊 Starting ECG data collection...")

        start_time = time.time()
        sample_count = 0

        try:
            while self.is_collecting and (time.time() - start_time) < self.duration:
                if self.serial_connection.in_waiting > 0:
                    # Read line from serial
                    line = self.serial_connection.readline().decode('utf-8').strip()

                    try:
                        # Parse ECG value (assuming comma-separated format: timestamp,ecg_value)
                        parts = line.split(',')
                        if len(parts) >= 2:
                            timestamp = float(parts[0])
                            ecg_value = float(parts[1])

                            # Add to queue
                            self.data_queue.put({
                                'timestamp': timestamp,
                                'ecg_signal': ecg_value,
                                'sample_id': sample_count
                            })

                            sample_count += 1

                            # Progress indicator
                            if sample_count % 100 == 0:
                                elapsed = time.time() - start_time
                                print(f"📈 Collected {sample_count} samples in {elapsed:.1f}s")

                        else:
                            # Single value format
                            ecg_value = float(line)
                            timestamp = time.time()

                            self.data_queue.put({
                                'timestamp': timestamp,
                                'ecg_signal': ecg_value,
                                'sample_id': sample_count
                            })

                            sample_count += 1

                    except ValueError as e:
                        # Skip invalid data points
                        continue

        except Exception as e:
            print(f"❌ Error during data collection: {e}")

        print(f"✅ Data collection completed! Total samples: {sample_count}")

    def start_collection(self):
        """Start ECG data collection"""
        if not self.connect_device():
            return None

        self.is_collecting = True

        # Start collection thread
        collection_thread = threading.Thread(target=self.collect_data_thread)
        collection_thread.start()

        # Wait for collection to complete
        collection_thread.join()

        # Close serial connection
        if self.serial_connection:
            self.serial_connection.close()

        # Collect all data from queue
        data = []
        while not self.data_queue.empty():
            data.append(self.data_queue.get())

        return data

    def save_to_csv(self, data, filename=None):
        """Save collected ECG data to CSV file"""
        if not data:
            print("❌ No data to save!")
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ecg_{timestamp}.csv"

        # Ensure raw_output directory exists
        output_dir = "raw_output"
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(filepath, index=False)

        print(f"💾 ECG data saved to: {filepath}")
        print(f"   - Samples: {len(df)}")
        print(f"   - Duration: {self.duration} seconds")
        print(".1f"
        return filepath

def list_available_ports():
    """List available serial ports"""
    import serial.tools.list_ports

    ports = serial.tools.list_ports.comports()
    if not ports:
        print("❌ No serial ports found!")
        print("💡 Make sure your ECG device is connected")
        return []

    print("🔍 Available serial ports:")
    for i, port in enumerate(ports):
        print(f"   {i+1}. {port.device} - {port.description}")

    return [port.device for port in ports]

def main():
    parser = argparse.ArgumentParser(description='Collect ECG Data from IoT Device')
    parser.add_argument('--port', default='COM3', help='Serial port (default: COM3)')
    parser.add_argument('--baudrate', type=int, default=9600, help='Baud rate (default: 9600)')
    parser.add_argument('--duration', type=int, default=30, help='Collection duration in seconds (default: 30)')
    parser.add_argument('--output', help='Output filename (optional)')
    parser.add_argument('--list-ports', action='store_true', help='List available serial ports')

    args = parser.parse_args()

    if args.list_ports:
        list_available_ports()
        return

    print("🫀 ECG Data Collector")
    print("=" * 40)

    # Create collector
    collector = ECGCollector(
        port=args.port,
        baudrate=args.baudrate,
        duration=args.duration
    )

    # Start collection
    data = collector.start_collection()

    if data:
        # Save to CSV
        filepath = collector.save_to_csv(data, args.output)

        if filepath:
            print("\n✅ ECG data collection completed successfully!")
            print(f"📁 File saved: {filepath}")
            print("\n💡 Next steps:")
            print("   1. Clean the data: python clean_ecg.py --input raw_output\\ecg_xxx.csv")
            print("   2. Make prediction: python predict_ecg.py --input cleaned_output\\cleaned_ecg_xxx.csv")
    else:
        print("❌ Failed to collect ECG data")

if __name__ == "__main__":
    main()# ============================================================
#  ECG Data Collector — AD8232 + Arduino Uno
#  With Real‑time Live Graph (Indefinite Recording)
#  Folder: ECG\collect_ecg.py
#
#  Usage:
#    python collect_ecg.py                      (run until stopped)
#    python collect_ecg.py --duration 120       (record for 120 seconds)
#    python collect_ecg.py --port COM3 --no-live (disable live graph)
# ============================================================

import serial
import serial.tools.list_ports
import csv
import os
import time
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ── Config ────────────────────────────────────────────────
BAUD_RATE    = 115200
SAMPLE_RATE  = 360        # Hz
DEFAULT_PORT = None
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_output")

# Live plot settings
WINDOW_SECONDS = 10        # show last 10 seconds of signal
MAX_POINTS = WINDOW_SECONDS * SAMPLE_RATE   # 3600 points

# ── Auto-detect Arduino port ───────────────────────────────
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = (port.description or "").lower()
        if any(k in desc for k in ["arduino", "ch340", "cp210", "usb serial", "uart"]):
            print(f"✅ Arduino auto-detected: {port.device} ({port.description})")
            return port.device
    if ports:
        print(f"⚠️  Arduino not detected. Using first port: {ports[0].device}")
        return ports[0].device
    return None

# ── Live plot class ────────────────────────────────────────
class LiveECGPlot:
    def __init__(self, max_points=MAX_POINTS, sample_rate=SAMPLE_RATE):
        self.max_points = max_points
        self.sample_rate = sample_rate
        self.data = deque(maxlen=max_points)
        self.time = deque(maxlen=max_points)
        self.start_time = None
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.line, = self.ax.plot([], [], 'b-', linewidth=0.8)
        self.ax.set_ylim(-10, 1050)    # ADC range 0-1023, plus margin
        self.ax.set_xlim(0, WINDOW_SECONDS)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('ECG Signal (ADC value)')
        self.ax.set_title('Real-time ECG Signal (0–1023)')
        self.ax.grid(True, alpha=0.3)
        plt.ion()
        plt.show(block=False)

    def update(self, value, elapsed_time):
        self.data.append(value)
        self.time.append(elapsed_time)
        if len(self.time) < 2:
            return
        current_time = elapsed_time
        xlim_left = max(0, current_time - WINDOW_SECONDS)
        self.ax.set_xlim(xlim_left, xlim_left + WINDOW_SECONDS)
        # Plot only visible points
        vis_time = [t for t in self.time if t >= xlim_left]
        vis_data = list(self.data)[-len(vis_time):]
        self.line.set_data(vis_time, vis_data)
        if len(vis_data) > 10:
            ymin = min(vis_data) - 20
            ymax = max(vis_data) + 20
            self.ax.set_ylim(max(0, ymin), min(1050, ymax))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)

# ── Main collector (indefinite mode) ───────────────────────
def collect_ecg(port, duration_sec=None, output_dir=None, live=True):
    """
    Collect ECG data. If duration_sec is None, run until user stops (Ctrl+C or close plot).
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"ecg_{timestamp}.csv")

    if duration_sec is not None:
        total_samples = duration_sec * SAMPLE_RATE
        print(f"⏱️  Recording for {duration_sec} seconds (~{total_samples} samples)")
    else:
        total_samples = None
        print("⏱️  Recording indefinitely (until stopped by Ctrl+C or closing graph)")

    print(f"\n📡 Connecting to {port} at {BAUD_RATE} baud...")
    print(f"💾 Output: {filename}")
    print("─" * 50)

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        time.sleep(2)
        ser.flushInput()
    except serial.SerialException as e:
        print(f"❌ Serial connection failed: {e}")
        return None

    live_plot = None
    if live:
        try:
            live_plot = LiveECGPlot()
            print("📈 Live graph enabled. Close the plot window to stop early.\n")
        except Exception as e:
            print(f"⚠️ Could not start live graph: {e}. Proceeding without graph.")
            live_plot = None

    start_time = time.time()
    sample_idx = 0
    collected = 0
    skipped_lines = 0

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sample_index", "ecg_value"])

        print("🔴 Recording started... (Press Ctrl+C or close graph window to stop)\n")

        try:
            while True:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except Exception:
                    skipped_lines += 1
                    continue

                if not line or line.startswith('#'):
                    continue

                try:
                    value = int(line)
                    writer.writerow([sample_idx, value])
                    sample_idx += 1
                    collected += 1

                    # Update live plot every ~0.2 sec (every 72 samples)
                    if live_plot and (collected % 72 == 0 or collected == 1):
                        elapsed = time.time() - start_time
                        live_plot.update(value, elapsed)
                        if not plt.fignum_exists(live_plot.fig.number):
                            print("\n⚠️ Graph window closed. Stopping recording.")
                            break

                    # Show progress every second (if duration given, else show elapsed time)
                    if collected % SAMPLE_RATE == 0:
                        elapsed = time.time() - start_time
                        if total_samples is not None:
                            pct = (collected / total_samples) * 100
                            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                            remaining = duration_sec - elapsed
                            print(f"\r[{bar}] {pct:.0f}% | {elapsed:.0f}s elapsed | {max(0, remaining):.0f}s left", end='', flush=True)
                            if collected >= total_samples:
                                break
                        else:
                            # Indefinite mode: just show elapsed time
                            print(f"\r🟢 Recording... {elapsed:.0f}s elapsed | {collected} samples", end='', flush=True)

                except ValueError:
                    skipped_lines += 1
                    continue

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Recording stopped by user (Ctrl+C).")

    ser.close()
    if live_plot:
        live_plot.close()

    elapsed_total = time.time() - start_time
    print(f"\n\n✅ Recording complete!")
    print(f"   Samples collected : {collected}")
    print(f"   Duration          : {elapsed_total:.1f}s")
    print(f"   Skipped lines     : {skipped_lines}")
    print(f"   File saved        : {filename}")
    print(f"\n📌 Next step: python clean_ecg.py --input \"{filename}\"")
    return filename


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Data Collector — AD8232 + Arduino")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (e.g. COM3). Auto-detect if not provided.")
    parser.add_argument("--duration", type=int, default=None,
                        help="Recording duration in seconds (default: indefinite until stopped)")
    parser.add_argument("--no-live", action="store_true",
                        help="Disable live graph (default: live graph enabled)")
    args = parser.parse_args()

    port = args.port or find_arduino_port() or DEFAULT_PORT
    if not port:
        print("❌ No serial port found. Connect Arduino and try again.")
        print("   Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"     {p.device} — {p.description}")
        exit(1)

    live = not args.no_live
    result = collect_ecg(port, duration_sec=args.duration, output_dir=OUTPUT_DIR, live=live)
    if result:
        print(f"\n✅ Done! File: {result}")