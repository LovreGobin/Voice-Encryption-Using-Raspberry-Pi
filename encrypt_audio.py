import pyaudio
import threading
from cryptography.fernet import Fernet
import os
import sys

# -----------------------------
# KEY MANAGEMENT
# -----------------------------
KEY_FILE = "key.key"
OUTPUT_FILE = "encrypted_audio.bin"

if not os.path.exists(KEY_FILE):
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print("Key generated and saved to key.key")
else:
    with open(KEY_FILE, "rb") as f:
        key = f.read()
    print("Key loaded from key.key")

cipher = Fernet(key)

# -----------------------------
# AUDIO SETTINGS
# -----------------------------
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

recording = False
frames = []

# -----------------------------
# FIND G430 MICROPHONE (Card 3)
# -----------------------------
p = pyaudio.PyAudio()

print("\n" + "="*60)
print("Available input devices:")
print("="*60)

input_device_index = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        print(f"  [{i}] {info['name']} ({info['maxInputChannels']} channels)")
        
        # Traži G430 ili Logitech G430
        if "g430" in info['name'].lower() or "logitech g430" in info['name'].lower():
            input_device_index = i
            print(f"       ← SELECTED (G430 Microphone)")

print("="*60)

if input_device_index is None:
    # Ako G430 nije pronađen, uzmi prvi dostupni
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            input_device_index = i
            print(f"\n G430 not found, using: {info['name']}")
            break

if input_device_index is None:
    print("\n ERROR: No input devices found!")
    p.terminate()
    sys.exit(1)

# -----------------------------
# RECORDING CONTROL
# -----------------------------
def keyboard_listener():
    global recording, frames
    while True:
        cmd = input().strip()
        if cmd == "1":
            recording = True
            frames = []
            print("RECORDING STARTED")
        elif cmd == "0":
            recording = False
            print("RECORDING STOPPED")
            save_encrypted_audio()

def save_encrypted_audio():
    if not frames:
        print("No audio recorded")
        return
    
    raw_audio = b"".join(frames)
    encrypted = cipher.encrypt(raw_audio)
    
    with open(OUTPUT_FILE, "wb") as f:
        f.write(encrypted)
    
    duration = len(raw_audio) / (RATE * CHANNELS * 2)
    print(f" Encrypted audio saved to {OUTPUT_FILE}")
    print(f" Size: {len(encrypted)} bytes")
    print(f" Duration: ~{duration:.2f} seconds")

# -----------------------------
# START RECORDING THREAD
# -----------------------------
threading.Thread(target=keyboard_listener, daemon=True).start()

# -----------------------------
# OPEN AUDIO STREAM
# -----------------------------
try:
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=CHUNK
    )
    print("\n Audio stream opened successfully")
except Exception as e:
    print(f"\ Error opening stream: {e}")
    p.terminate()
    sys.exit(1)

print("\n" + "="*60)
print("CONTROLS:")
print("  Press 1 + ENTER to START recording")
print("  Press 0 + ENTER to STOP & ENCRYPT")
print("  Press Ctrl+C to exit")
print("="*60 + "\n")

# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        if recording:
            frames.append(data)
except KeyboardInterrupt:
    print("\n\n Exiting...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
