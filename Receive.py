#!/usr/bin/env python3
import numpy as np
import sys
import os
import termios
import tty

# =============================
# CONFIG
# =============================
IQ_FILE = "received_signal.iq"
OUTPUT_PREFIX = "received_dump"

SAMPLE_RATE = 2_000_000
BAUD_RATE = 250_000

# PREAMBLE = 0xAA x 4  = 10101010 * 4
PREAMBLE_BITS = [1, 0, 1, 0, 1, 0, 1, 0] * 4
# START    = 0x7E x 2  = 01111110 * 2
START_BITS    = [0, 1, 1, 1, 1, 1, 1, 0] * 2

# =============================
# TERMINAL RAW INPUT
# =============================
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

def read_key():
    return sys.stdin.read(1)

# =============================
# UTILS
# =============================
def crc8(data: bytes) -> int:
  
    return sum(data) & 0xFF

def bits_to_bytes(bits: np.ndarray) -> bytes:


    usable_len = (len(bits) // 8) * 8
    bits = bits[:usable_len]

    out = []
    for i in range(0, len(bits), 8):
        b = 0
        for j in range(8):
            b = (b << 1) | int(bits[i + j])
        out.append(b)
    return bytes(out)

def find_sequence(bits: np.ndarray, pattern: list[int], start: int = 0) -> int:
 
    plen = len(pattern)
    if plen == 0:
        return -1

    for i in range(start, len(bits) - plen + 1):
        if bits[i:i+plen].tolist() == pattern:
            return i
    return -1

def load_iq() -> np.ndarray | None:

    if not os.path.exists(IQ_FILE):
        return None

    raw = np.fromfile(IQ_FILE, dtype=np.int8)
    if len(raw) < 2:
        return None

    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    
    samples = (i + 1j * q) / 128.0
    return samples.astype(np.complex64)

def fsk_demod(samples: np.ndarray) -> np.ndarray:
    
    spb = int(SAMPLE_RATE / BAUD_RATE)  
    bits = []

    if spb <= 0:
        return np.array([], dtype=np.uint8)

    for idx in range(0, len(samples) - spb, spb):
        chunk = samples[idx:idx + spb]
       
        phase = np.unwrap(np.angle(chunk))
    
        freq = np.mean(np.diff(phase))
        bits.append(1 if freq > 0 else 0)

    return np.array(bits, dtype=np.uint8)

def parse_packets(bits: np.ndarray) -> dict[int, bytes]:

    packets: dict[int, bytes] = {}
    cursor = 0

    while True:
      
        p = find_sequence(bits, PREAMBLE_BITS, cursor)
        if p == -1:
            break

  
        s = find_sequence(bits, START_BITS, p + len(PREAMBLE_BITS))
        if s == -1:
            break

        ptr = s + len(START_BITS)

        
        hdr_bits = bits[ptr:ptr + 16]
        if len(hdr_bits) < 16:
            break
        hdr = bits_to_bytes(hdr_bits)
        if len(hdr) < 2:
            break

        pid, total = hdr
        ptr += 16

  
        length_bits = bits[ptr:ptr + 16]
        if len(length_bits) < 16:
            break
        length_bytes = bits_to_bytes(length_bits)
        if len(length_bytes) < 2:
            break

        length = int.from_bytes(length_bytes, "big")
        ptr += 16

     
        needed_payload_bits = length * 8
        payload_bits = bits[ptr:ptr + needed_payload_bits]
        if len(payload_bits) < needed_payload_bits:
            break
        payload = bits_to_bytes(payload_bits)
        ptr += needed_payload_bits

      
        crc_bits = bits[ptr:ptr + 8]
        if len(crc_bits) < 8:
            break
        crc_val = bits_to_bytes(crc_bits)[0]
        ptr += 8


        if crc8(payload) == crc_val:
            packets[pid] = payload


        cursor = ptr

    return packets

# =============================
# MAIN LOOP
# =============================
print("RX running")
print("Press '1' to process IQ and dump .bin")
print("Press 'q' to quit")

dump_id = 0

try:
    while True:
        key = read_key()

        if key == "1":
            print("\nProcessing IQ...")
            samples = load_iq()

            if samples is None:
                print("No IQ data found or file is too short.")
                continue

            print(f"Loaded {len(samples)} complex samples.")
            bits = fsk_demod(samples)

            if bits.size == 0:
                print("No bits demodulated (check SAMPLE_RATE / BAUD_RATE).")
                continue

            print(f"Demodulated {len(bits)} bits. Parsing packets...")
            packets = parse_packets(bits)

            if not packets:
                print("No valid packets found (preamble/start/CRC mismatch).")
                continue

            ordered_ids = sorted(packets.keys())
            data = b"".join(packets[i] for i in ordered_ids)

            dump_id += 1
            out = f"{OUTPUT_PREFIX}_{dump_id}.bin"

            with open(out, "wb") as f:
                f.write(data)

            print(f"Wrote {out} ({len(data)} bytes)")
            print(f"Packets received: {len(ordered_ids)} (ids: {ordered_ids})")

        elif key == "q":
            print("\nExiting RX.")
            break

finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
