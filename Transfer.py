#!/usr/bin/env python3

import numpy as np
import os
import sys
import subprocess

# =============================
# CONFIG
# =============================


ENCRYPTED_FILE = "encrypted_audio.bin"

CENTER_FREQ = 433.92e6
SAMPLE_RATE = 2_000_000
TX_GAIN = 40
BANDWIDTH = 1_750_000

BAUD_RATE = 250_000
FREQ_SEP = 100e3

MAX_PAYLOAD = 512  # bytes per packet
PREAMBLE = bytes([0xAA, 0xAA, 0xAA, 0xAA])
START = bytes([0x7E, 0x7E])

TX_SERIAL = "000000000000000075b068dc322e9f07"
IQ_FILE = "tx_temp.iq"

# =============================
# LOAD DATA
# =============================
if not os.path.exists(ENCRYPTED_FILE):
    print("ERROR: encrypted_audio.bin not found")
    sys.exit(1)

with open(ENCRYPTED_FILE, "rb") as f:
    encrypted_data = f.read()

print(f"Loaded encrypted file: {len(encrypted_data)} bytes")

# =============================
# UTILS
# =============================
def split_chunks(data, size):
    return [data[i:i+size] for i in range(0, len(data), size)]

def crc8(data):
    return sum(data) & 0xFF

def bytes_to_bits(data):
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    return np.array(bits, dtype=np.uint8)

def fsk_modulate(bits, sample_rate, baud_rate, freq_sep):
    samples_per_bit = int(sample_rate / baud_rate)
    t = np.arange(samples_per_bit, dtype=np.float32) / sample_rate

    symbols = []
    for bit in bits:
        f = freq_sep / 2 if bit else -freq_sep / 2
        symbol = np.exp(2j * np.pi * f * t)
        symbols.append(symbol)

    sig = np.concatenate(symbols).astype(np.complex64)

    # normalizacija na |max| � 1
    sig /= np.max(np.abs(sig)) * 1.2
    return sig

def complex_to_int8_interleaved(iq_complex: np.ndarray) -> np.ndarray:

    i = np.clip(np.real(iq_complex) * 127.0, -128, 127).astype(np.int8)
    q = np.clip(np.imag(iq_complex) * 127.0, -128, 127).astype(np.int8)

    interleaved = np.empty(i.size * 2, dtype=np.int8)
    interleaved[0::2] = i
    interleaved[1::2] = q
    return interleaved

# =============================
# FRAME FORMAT
# =============================
def create_packet(packet_id, total_packets, payload):
    length = len(payload).to_bytes(2, "big")

    header = bytes([
        packet_id & 0xFF,
        total_packets & 0xFF
    ])

    frame = (
        PREAMBLE +
        START +
        header +
        length +
        payload +
        bytes([crc8(payload)])
    )
    return frame

# =============================
# PACKETIZE
# =============================
packets = split_chunks(encrypted_data, MAX_PAYLOAD)
total_packets = len(packets)

print(f"Total packets: {total_packets}")

# =============================
# TRANSMIT LOOP
# =============================
print("\nRF TRANSMISSION READY")
input("Press ENTER to start transmission...")

for i, payload in enumerate(packets):
    print(f"Sending packet {i+1}/{total_packets}")

    frame = create_packet(i, total_packets, payload)
    bits = bytes_to_bits(frame)

    # 1) FSK modulacija u complex64
    iq_complex = fsk_modulate(bits, SAMPLE_RATE, BAUD_RATE, FREQ_SEP)

    # 2) KONVERZIJA ZA HACKRF: complex64 -> int8 interleaved I/Q
    iq_int8 = complex_to_int8_interleaved(iq_complex)

    # 3) Zapis u .iq datoteku u formatu koji hackrf_transfer ocekuje
    iq_int8.tofile(IQ_FILE)

    cmd = [
        "hackrf_transfer",
        "-d", TX_SERIAL,
        "-t", IQ_FILE,
        "-f", str(int(CENTER_FREQ)),
        "-s", str(int(SAMPLE_RATE)),
        "-x", str(int(TX_GAIN)),
        "-b", str(int(BANDWIDTH)),
        "-a", "1",
    ]

    subprocess.run(cmd, check=True)

print("\nTransmission complete.")
