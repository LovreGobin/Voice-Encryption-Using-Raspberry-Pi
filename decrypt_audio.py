import pyaudio
from cryptography.fernet import Fernet, InvalidToken
import os
import sys

# -----------------------------
# CONFIGURATION
# -----------------------------
KEY_FILE = "key.key"
ENCRYPTED_FILE = "encrypted_audio.bin"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# -----------------------------
# LOAD KEY
# -----------------------------
print("="*60)
print("AUDIO DECRYPTION & PLAYBACK")
print("="*60)

key_input = input("\nPaste the key or press ENTER to load from key.key: ").strip()

if key_input == "":
    if not os.path.exists(KEY_FILE):
        print("Key file not found!")
        sys.exit(1)
    with open(KEY_FILE, "rb") as f:
        key = f.read()
    print("Key loaded from key.key")
else:
    key = key_input.encode()
    print("Using provided key")

cipher = Fernet(key)

# -----------------------------
# LOAD & DECRYPT AUDIO
# -----------------------------
if not os.path.exists(ENCRYPTED_FILE):
    print(f"Encrypted audio file '{ENCRYPTED_FILE}' not found!")
    sys.exit(1)

with open(ENCRYPTED_FILE, "rb") as f:
    encrypted_data = f.read()

print(f"Loaded encrypted file ({len(encrypted_data)} bytes)")

try:
    decrypted_audio = cipher.decrypt(encrypted_data)
    duration = len(decrypted_audio) / (RATE * CHANNELS * 2)
    print(f"Decryption successful ({len(decrypted_audio)} bytes)")
    print(f"Audio duration: ~{duration:.2f} seconds")
except InvalidToken:
    print("WRONG KEY! Audio remains encrypted.")
    sys.exit(1)

# -----------------------------
# FIND PRO X HEADSET (Card 2)
# -----------------------------
p = pyaudio.PyAudio()

print("\n" + "="*60)
print("Available output devices:")
print("="*60)

output_device_index = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxOutputChannels"] > 0:
        print(f"  [{i}] {info['name']} ({info['maxOutputChannels']} channels)")
        
        # Traži PRO X
        if "pro x" in info['name'].lower():
            output_device_index = i
            print(f"       SELECTED (PRO X Headset)")

print("="*60)

# Ako korisnik želi ručno odabrati
manual_select = input("\nUse PRO X or select manually? (ENTER for PRO X / number for manual): ").strip()

if manual_select.isdigit():
    device_index = int(manual_select)
    info = p.get_device_info_by_index(device_index)
    if info["maxOutputChannels"] > 0:
        output_device_index = device_index
        print(f"Manually selected: {info['name']}")
    else:
        print(f"Device {device_index} has no output channels!")
        p.terminate()
        sys.exit(1)
elif output_device_index is None:
    print("PRO X not found and no manual selection made!")
    p.terminate()
    sys.exit(1)
else:
    info = p.get_device_info_by_index(output_device_index)
    print(f"Using PRO X: {info['name']}")

# -----------------------------
# PLAY AUDIO
# -----------------------------
print("\n" + "="*60)
print("Starting playback on PRO X headset...")
print("="*60)

try:
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        output_device_index=output_device_index
    )
    
    # Play audio in chunks
    print("Playing...")
    for i in range(0, len(decrypted_audio), CHUNK):
        chunk = decrypted_audio[i:i+CHUNK]
        stream.write(chunk)
    
    stream.stop_stream()
    stream.close()
    print("Playback finished")
    
except Exception as e:
    print(f"Playback error: {e}")
finally:
    p.terminate()

print("="*60)
