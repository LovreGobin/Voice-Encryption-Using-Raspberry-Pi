# Voice Encryption using Raspberry Pi Microcontroller

**University of Split**  
**Faculty of Electrical Engineering, Mechanical Engineering and Naval Architecture**  
**Computer Engineering (222)**

**Course:** Wireless Network Security

**Authors:** Mateo Franjić, Lovre Gobin, Toni Mišić

**Split, February 2026**

---

## Introduction

The main goal of the project was to develop a system for encrypted voice communication based on the Raspberry Pi platform. The system encompasses voice signal recording, its digitization, encryption using a selected cryptographic algorithm, and subsequent decryption and sound reproduction on the receiving side.

The project also includes an extended task in which encrypted data is transmitted via radio waves between two HackRF devices. This part serves as a demonstration of wireless transmission capabilities of already encrypted voice signals over an RF channel.

The report is organized so that the initial chapter lists the equipment used. This is followed by a chapter that describes in detail the process of encryption and decryption of voice signals on the Raspberry Pi, including the software libraries used and the results achieved. After that, the implementation and testing of wireless transmission of encrypted data using two HackRF devices is covered, describing the procedures for sending and receiving data as well as analysis of the obtained results.

---

## Equipment Used

- **Raspberry Pi 5**
- **2 Headsets**
- **2 USB Adapters** (Logitech G435 and Pro X)
- **2 HackRF devices**

For system implementation, the Raspberry Pi 5 was used as the central unit, two headsets, two USB adapters (Logitech G435 and Pro X), and two HackRF devices. The Raspberry Pi 5 runs the official Raspberry Pi OS operating system, and the application logic for voice encryption and decryption is implemented in the Python programming language.

For data encryption, the **Fernet** module from the `cryptography` library was used, which is based on symmetric cryptography: it uses **AES-128** in CBC mode for encryption and **HMAC-SHA256** for message integrity and authenticity verification. This ensures that encrypted content cannot be read or modified without possessing the correct secret key.

Both headsets are connected via their respective USB adapters to the Raspberry Pi. The microphone of the first headset is used to record the voice signal, which is then digitized and encrypted on the Raspberry Pi using the Fernet key. If the same secret key is used for decryption, the encrypted data can be successfully decrypted, and the result is reproduced on the second headset, where the user hears the original voice signal in an understandable form (Figure 2.1).

<img width="975" height="78" alt="image" src="https://github.com/user-attachments/assets/b64602b2-93df-417b-97f4-0e3cd35b1a20" /><br>
Flow diagram for voice encryption and decryption

In the extended part of the system, encrypted data is additionally transmitted via radio waves using two HackRF devices. On the transmitting side, the encrypted message is modulated and sent from the first HackRF device to the selected RF frequency. On the receiving side, the second HackRF device receives and demodulates the signal and reconstructs the encrypted data from the obtained binary stream. Decryption is then attempted on the obtained data using the same Fernet key as on the transmitting side; in case of successful decryption, the original voice signal is obtained, which can be reproduced on the receiving headset.

---

## Voice Encryption and Decryption

### Encryption

**File: `encrypt_audio.py`**

At the beginning of the program, modules are added: `pyaudio` for working with audio input, `threading` for managing parallel execution, and `Fernet` from the `cryptography` library for symmetric data encryption. The program stores the secret key in a `key.key` file for easier testing. If the file does not exist, a new symmetric key is generated using the `Fernet.generate_key()` function and written to the file, while otherwise the key is loaded from the existing file. Based on the loaded key, a `cipher` object of the Fernet class is created, which is later used for audio data encryption.

```python
# Lines 1-20: Import modules and key management
import pyaudio
import threading
from cryptography.fernet import Fernet
import os
import sys

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
```

Audio recording is configured with buffer size `CHUNK = 1024`, sample format `pyaudio.paInt16` (16-bit integer), one channel (mono), and a sampling frequency of 16 kHz since the maximum voice frequency is 8kHz (sampling theorem). These parameters define how PyAudio will read samples from the input device. The `frames` variable is used to temporarily store blocks (chunks) of raw audio signal while recording is active. Using the `PyAudio()` object, the program goes through all available input devices and tries to find the Logitech G430 microphone by name. If this device is not found, the first available audio input device is selected.

```python
# Lines 1-24: Audio configuration and device selection
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

recording = False
frames = []

p = pyaudio.PyAudio()

input_device_index = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        if "g430" in info['name'].lower() or "logitech g430" in info['name'].lower():
            input_device_index = i

if input_device_index is None:
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            input_device_index = i
            break
```

Recording management is implemented via a separate thread that reads keyboard input in the `keyboard_listener` function. If the user enters `1`, the global `recording` flag is set to `True` and recording of a new audio clip begins, while entering `0` stops recording and starts the `save_encrypted_audio()` function. This way, recording can be started and stopped without interrupting the main loop that continuously reads audio data.

The `save_encrypted_audio()` function first joins all collected blocks into a single byte array `raw_audio`. The obtained raw audio data is then encrypted by calling `cipher.encrypt(raw_audio)`, applying symmetric encryption to the entire recorded signal. The encrypted data is saved to the `encrypted_audio.bin` file, which later serves as input for decryption and sound reproduction.

```python
# Lines 1-21: Keyboard listener and encryption function
def keyboard_listener():
    global recording, frames
    while True:
        cmd = input().strip()
        if cmd == "1":
            recording = True
            frames = []
        elif cmd == "0":
            recording = False
            save_encrypted_audio()

def save_encrypted_audio():
    if not frames:
        return

    raw_audio = b"".join(frames)
    encrypted = cipher.encrypt(raw_audio)

    with open(OUTPUT_FILE, "wb") as f:
        f.write(encrypted)
```

The `keyboard_listener` function is started in a separate thread so that audio can be processed and user input can be reacted to simultaneously, without blocking the main program. After that, an audio stream is opened to collect sound using the `p.open` method, using the previously defined format parameters, number of channels, sampling frequency, and selected input device.

In the main loop, the program continuously reads blocks of audio data from the stream using the `stream.read(CHUNK)` function. If the `recording` flag is enabled, the read block is added to the `frames` list. This way, all consecutive audio signal blocks are collected during recording, which are then joined and encrypted into a single binary file when recording is stopped.

```python
# Lines 1-16: Main recording loop
threading.Thread(target=keyboard_listener, daemon=True).start()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=input_device_index,
    frames_per_buffer=CHUNK
)

while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    if recording:
        frames.append(data)
```

---

### Decryption

**File: `decrypt_audio.py`**

At the beginning of the decryption program, the same audio parameters (`FORMAT`, `CHANNELS`, `RATE`, `CHUNK`) are defined so that the playback configuration matches the recording configuration. The user is given the option to enter the secret key directly via keyboard or to have the key automatically loaded from the `key.key` file. In both cases, a Fernet object is created with the selected key, which will be used to decrypt the previously encrypted audio recording.

```python
# Lines 1-20: Import modules and key loading
import pyaudio
from cryptography.fernet import Fernet, InvalidToken
import os
import sys

KEY_FILE = "key.key"
ENCRYPTED_FILE = "encrypted_audio.bin"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

key_input = input("Paste the key or press ENTER to load from key.key: ").strip()
if key_input == "":
    with open(KEY_FILE, "rb") as f:
        key = f.read()
else:
    key = key_input.encode()

cipher = Fernet(key)
```

The program then checks if the `encrypted_audio.bin` file exists and, if it exists, loads it into memory as a byte array. The `cipher.decrypt(encrypted_data)` function attempts to decrypt the data using the given Fernet key; in case the key is incorrect or the data has been modified, an `InvalidToken` exception is thrown, which prevents further processing of incorrect or compromised content. If decryption is successful, an array of raw audio samples is obtained, from whose length the approximate duration of the audio recording is calculated.

```python
# Lines 1-11: File loading and decryption
if not os.path.exists(ENCRYPTED_FILE):
    sys.exit(1)

with open(ENCRYPTED_FILE, "rb") as f:
    encrypted_data = f.read()

try:
    decrypted_audio = cipher.decrypt(encrypted_data)
    duration = len(decrypted_audio) / (RATE * CHANNELS * 2)
except InvalidToken:
    sys.exit(1)
```

For playback of decrypted sound, the PyAudio library is used. The program first goes through the list of available output devices and tries to automatically find the PRO X headset by device name. The user is given the option to manually select the output device by entering the index.

```python
# Lines 1-10: Output device selection
p = pyaudio.PyAudio()
output_device_index = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxOutputChannels"] > 0 and "pro x" in info["name"].lower():
        output_device_index = i

manual_select = input("Use PRO X or select manually? ").strip()
if manual_select.isdigit():
    output_device_index = int(manual_select)
```

For playback, an output audio stream is opened using the `p.open` method, again using the same format, channel, and sampling frequency parameters as during recording. The decrypted audio recording is divided into blocks of size `CHUNK`, which are sequentially sent to the output stream by calling `stream.write(chunk)`. This way, the user hears the original voice signal on the selected headset, provided that the same secret key was used for decryption as in the encryption phase.

```python
# Lines 1-15: Audio playback
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    output=True,
    output_device_index=output_device_index
)

for i in range(0, len(decrypted_audio), CHUNK):
    chunk = decrypted_audio[i:i+CHUNK]
    stream.write(chunk)

stream.stop_stream()
stream.close()
p.terminate()
```

---

### Results

The functionality of the system (Figure 3.1) was tested by first encrypting and saving the recorded voice signal to a file, then transferring it to another device where the decryption and playback program was running. When using the correct Fernet key, the encrypted audio recording was successfully decrypted into the original sample array and reproduced on the selected output audio device (headset). In all tests performed, the message after decryption was clearly understandable, without any noticeable distortions or interruptions in playback, which confirms the correct operation of the implemented voice communication encryption and decryption system.

<img width="783" height="587" alt="image" src="https://github.com/user-attachments/assets/942d96de-c65e-42e3-9f0b-9eb172522958" /><br>
System for voice encryption and decryption

Figures 3.2 and 3.3 show console output when running encryption and decryption.

<img width="592" height="543" alt="image" src="https://github.com/user-attachments/assets/7171abab-60ea-42e4-990d-d7921d7facff" /><br>
Console output during encryption

<img width="598" height="569" alt="image" src="https://github.com/user-attachments/assets/ffa6a990-96fd-46fb-b722-8a458bf73510" /><br>
Console output during decryption

---

## Sending and Receiving Encrypted Data via Radio Waves

### Sending Encrypted Data and Modulation

**File: `tx_encrypted.py`**

At the beginning of the transmitter program, RF transmission parameters are defined, including center frequency `CENTER_FREQ = 433.92 MHz`, sampling rate `SAMPLE_RATE = 2 MS/s`, transmitter gain `TX_GAIN`, and bandwidth `BANDWIDTH`. Additionally, symbol transmission rate `BAUD_RATE = 250 kbaud` and frequency spacing `FREQ_SEP` for FSK modulation are set. The maximum user payload size in one packet is limited to 512 bytes (`MAX_PAYLOAD`), and a preamble (multiple consecutive 0xAA bytes) and start sequence 0x7E 0x7E are placed in the frame header for more reliable frame start recognition on the receiving side. `TX_SERIAL` represents the serial number of the transmitter HackRF device. The program loads the encrypted audio recording from the `encrypted_audio.bin` file, which contains data obtained in the previous voice encryption phase.

```python
# Lines 1-22: Import modules and parameter configuration
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

if not os.path.exists(ENCRYPTED_FILE):
    sys.exit(1)

with open(ENCRYPTED_FILE, "rb") as f:
    encrypted_data = f.read()
```

The encrypted data is first divided into smaller parts of maximum size 512 bytes by the `split_chunks` function, resulting in an array of packets that are sent separately over the RF channel. Each packet receives a unique identifier (`packet_id`) and information about the total number of packets (`total_packets`), which is stored in the header so the receiver can reconstruct the original data order. A two-byte length field and a simple CRC8 checksum (calculated as the sum of all bytes modulo 256) are additionally placed in the frame for payload integrity verification. The final frame format consists of preamble, start sequence, header with identifier and packet number, length field, payload, and final CRC byte.

```python
# Lines 1-33: Packet creation functions
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

packets = split_chunks(encrypted_data, MAX_PAYLOAD)
total_packets = len(packets)
```

During transmission, each frame is first converted from a byte array to a bit array by the `bytes_to_bits` function. These bits then enter the `fsk_modulate` function, which implements binary FSK modulation: for a bit value of 1, a complex sinusoidal signal with positive frequency offset +FREQ_SEP/2 is generated, and for bit 0 with negative offset −FREQ_SEP/2. For each bit, a corresponding number of samples `samples_per_bit = SAMPLE_RATE / BAUD_RATE` is generated, and all symbols are joined into a single complex array `sig`. The signal is normalized so that the maximum amplitude does not exceed the specified value, thereby avoiding transmitter saturation.

Since HackRF expects IQ samples in interleaved signed int8 format (I and Q components alternately in the range from −128 to 127), the `complex_to_int8_interleaved` function scales the real and imaginary parts of the complex signal to the appropriate range and places them in a common array. The thus prepared IQ samples are written to an `.iq` file that serves as an input data stream for `hackrf_transfer`.

```python
# Lines 1-22: FSK modulation and IQ conversion
def fsk_modulate(bits, sample_rate, baud_rate, freq_sep):
    samples_per_bit = int(sample_rate / baud_rate)
    t = np.arange(samples_per_bit, dtype=np.float32) / sample_rate

    symbols = []
    for bit in bits:
        f = freq_sep / 2 if bit else -freq_sep / 2
        symbol = np.exp(2j * np.pi * f * t)
        symbols.append(symbol)

    sig = np.concatenate(symbols).astype(np.complex64)
    sig /= np.max(np.abs(sig)) * 1.2
    return sig

def complex_to_int8_interleaved(iq_complex: np.ndarray) -> np.ndarray:
    i = np.clip(np.real(iq_complex) * 127.0, -128, 127).astype(np.int8)
    q = np.clip(np.imag(iq_complex) * 127.0, -128, 127).astype(np.int8)

    interleaved = np.empty(i.size * 2, dtype=np.int8)
    interleaved[0::2] = i
    interleaved[1::2] = q
    return interleaved
```

In the main transmitter loop, all prepared packets are processed. For each packet, the `create_packet` function is called, which builds the entire frame with preamble, header, length field, payload, and CRC. The resulting frame is converted to a bit array, then FSK-modulated into a complex IQ signal and converted to interleaved int8 IQ format suitable for HackRF. The IQ data is saved to a temporary file, and then sent to the HackRF device at the specified frequency, sampling rate, and gain using `hackrf_transfer`. This way, encrypted audio data is transmitted over the RF channel in the form of packetized FSK signal.

```python
# Lines 1-20: Main transmission loop
for i, payload in enumerate(packets):
    frame = create_packet(i, total_packets, payload)
    bits = bytes_to_bits(frame)

    iq_complex = fsk_modulate(bits, SAMPLE_RATE, BAUD_RATE, FREQ_SEP)
    iq_int8 = complex_to_int8_interleaved(iq_complex)
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
```

---

### Receiving Encrypted Data and Demodulation

**File: `rx_encrypted.py`**

For signal reception, `hackrf_transfer` is used, which records IQ samples from the HackRF One device to a binary file `received_signal.iq`. The parameter `-r received_signal.iq` specifies that the device operates in receive mode (RX) and saves received samples to the specified file. The option `-f 433920000` sets the receiver operating frequency to 433.92 MHz, which must match the center frequency at which the transmitter sends the modulated FSK signal. The parameter `-s 2000000` defines a sampling rate of 2 MS/s, aligned with the transmitter configuration to preserve signal shape information. Receiver gain is adjusted using options `-l 40` and `-g 20`, which control LNA (IF) and VGA (baseband) gain at the HackRF device input. This way, weaker signals can be compensated or noise can be reduced, depending on transmission conditions and distance between transmitter and receiver. The parameter `-d 0000000000000000675c62dc322b36cf` selects the HackRF device by serial number.

```bash
# Command line: HackRF receiver configuration
hackrf_transfer -r received_signal.iq -f 433920000 -s 2000000 -l 40 -g 20 -d 0000000000000000675c62dc322b36cf
```

The script on the receiving side takes IQ samples recorded by the HackRF device, performs simple 2-FSK demodulation, recognizes frames by preamble and start sequence, and extracts the payload from each valid frame. Valid packets are joined into the original byte array, which is saved to a binary file and represents the reconstruction of the original encrypted message.

At the beginning, file names are defined (`IQ_FILE` for input IQ samples and `OUTPUT_PREFIX` for output binary files) and basic signal parameters: sampling rate `SAMPLE_RATE = 2 MS/s` and symbol transmission rate `BAUD_RATE = 250 kbaud`, which must be aligned with the transmitter configuration. Preamble and start sequence are defined as bit arrays: preamble corresponds to four consecutive 0xAA bytes (pattern 10101010 repeated four times), while the start sequence represents two 0x7E bytes (binary pattern 01111110 repeated twice). These patterns are later used to recognize the frame start in the demodulated bit stream.

```python
# Lines 1-10: Configuration and constants
IQ_FILE = "received_signal.iq"
OUTPUT_PREFIX = "received_dump"

SAMPLE_RATE = 2_000_000
BAUD_RATE = 250_000


PREAMBLE_BITS = [1, 0, 1, 0, 1, 0, 1, 0] * 4

START_BITS    = [0, 1, 1, 1, 1, 1, 1, 0] * 2
```

The `load_iq()` function loads recorded IQ samples from the `received_signal.iq` file in the form of 8-bit signed integers, where I and Q components are interleaved (alternating). The real (I) and imaginary (Q) parts are extracted from the array, which are then scaled to an approximate range from −1 to 1, resulting in a complex signal `samples` of type complex64 suitable for further processing.

```python
# Lines 1-12: IQ sample loading function
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
```

The `fsk_demod` function performs simple 2-FSK demodulation by dividing the complex signal into blocks of length `samples_per_bit = SAMPLE_RATE / BAUD_RATE`, where each block corresponds to one bit. For each block, the sample phase is calculated, the phase is unwrapped, and the average phase difference is used as a frequency approximation. If the average frequency is positive, the bit is interpreted as 1, otherwise as 0. The result is a bit array representing the demodulated digital signal.

```python
# Lines 1-15: FSK demodulation function
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

```

The `bits_to_bytes` function converts a bit array (values 0 and 1) into a byte array. It first cuts the array length to the nearest lower multiple of 8, then assembles each block of 8 bits into one byte in MSB-first order. This way, the demodulated bit stream is prepared for further interpretation of frame header, length, and payload.

```python
# Lines 1-12: Bit to byte conversion
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

```

Preamble and start sequence are defined as known bit arrays: preamble corresponds to four consecutive 0xAA bytes (pattern 10101010), and start sequence to two 0x7E bytes (01111110). The `find_sequence` function searches the demodulated bit stream and tries to find the given pattern starting from the specified position. This mechanism allows the receiver to reliably detect the frame start within the continuous bit stream.

```python
# Lines 1-13: Pattern recognition function
PREAMBLE_BITS = [1, 0, 1, 0, 1, 0, 1, 0] * 4
START_BITS    = [0, 1, 1, 1, 1, 1, 1, 0] * 2

def find_sequence(bits: np.ndarray, pattern: list[int], start: int = 0) -> int:
    plen = len(pattern)
    if plen == 0:
        return -1

    for i in range(start, len(bits) - plen + 1):
        if bits[i:i+plen].tolist() == pattern:
            return i
    return -1

```

The `parse_packets` function searches the demodulated bit stream and looks for frames with the specified structure: preamble, start sequence, header, length, payload, and CRC. After finding the preamble and start sequence, 16 header bits are read and converted to two bytes: packet identifier (`packet_id`) and total number of packets (`total_packets`). The next 16 bits represent the length field, which is interpreted as the number of payload bytes. After that, the parser reads as many bits as specified by the length field and converts them into a `payload` byte array.

At the very end of the frame, 8 CRC bits are read, which are compared with the value obtained by the `crc8(payload)` function, where CRC is calculated as the sum of all payload bytes modulo 256. If the CRC values match, the packet is considered valid and stored in `packets` under its identifier key. This ensures that only packets with verified integrity are used in further processing.

```python
# Lines 1-44: Packet parsing function
def crc8(data: bytes) -> int:
    return sum(data) & 0xFF

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
        hdr = bits_to_bytes(hdr_bits)
        pid, total = hdr
        ptr += 16

        length_bits = bits[ptr:ptr + 16]
        length_bytes = bits_to_bytes(length_bits)
        length = int.from_bytes(length_bytes, "big")
        ptr += 16

        needed_payload_bits = length * 8
        payload_bits = bits[ptr:ptr + needed_payload_bits]
        payload = bits_to_bytes(payload_bits)
        ptr += needed_payload_bits

        crc_bits = bits[ptr:ptr + 8]
        crc_val = bits_to_bytes(crc_bits)[0]
        ptr += 8

        if crc8(payload) == crc_val:
            packets[pid] = payload

        cursor = ptr

    return packets

```

In the main receiver loop, the program waits for keyboard input. When key `1` is pressed, processing of currently available IQ samples is started: loading from file, FSK demodulation, and packet parsing. If bits are successfully demodulated and valid packets are found, they are sorted by identifier (`packet_id`) and their payload is joined into a single byte array `data`.

The resulting byte array is written to a file named `received_dump_N.bin`, where index N increments for each new reception. This file represents the reconstructed encrypted audio recording obtained by RF transmission, which can then be passed to the decryption script.

```python
# Lines 1-33: Main receiver loop
print("RX running")
print("Press '1' to process IQ and dump .bin")
print("Press 'q' to quit")

dump_id = 0

while True:
    key = read_key()

    if key == "1":
        samples = load_iq()
        if samples is None:
            continue

        bits = fsk_demod(samples)
        if bits.size == 0:
            continue

        packets = parse_packets(bits)
        if not packets:
            continue

        ordered_ids = sorted(packets.keys())
        data = b"".join(packets[i] for i in ordered_ids)

        dump_id += 1
        out = f"{OUTPUT_PREFIX}_{dump_id}.bin"
        with open(out, "wb") as f:
            f.write(data)

    elif key == "q":
        break

```

---

### RF Results

In testing with both HackRFs on the same laptop, the system managed to achieve transmission and reception of packetized encrypted data (Figure 4.1), but final decryption was unsuccessful due to packet loss or damage in the RF channel.

<img width="756" height="567" alt="image" src="https://github.com/user-attachments/assets/48d3f1e2-d74a-4dff-9a4f-bfeb98b3c61a" /><br>
System with two HackRF devices

Since symmetric Fernet encryption is used at a higher level, which includes HMAC for integrity and authenticity verification of encrypted data in addition to AES encryption, any even smallest change in bytes in the encrypted message causes decryption to return an `InvalidToken` error and completely refuse. Therefore, even when most of the encrypted audio recording is successfully transmitted and only individual packets are missing or partially damaged, Fernet does not allow partial decryption nor deliver partially decoded content, but protects message integrity by blocking any output. Console outputs during data sending and receiving can be seen in Figure 4.2.

<img width="975" height="439" alt="image" src="https://github.com/user-attachments/assets/5ad5bfdd-ec62-4598-8a1f-8268681de98e" /><br>
Console outputs during sending and receiving

---

## Conclusion

The conducted project showed that it is possible to create a functional system for voice communication encryption on the Raspberry Pi platform using symmetric Fernet encryption, which combines AES-128 with the HMAC-SHA256 mechanism for message integrity and authenticity verification. In a scenario where the encrypted audio recording is locally saved and immediately decrypted on the same device or by sending the encrypted message securely to another device, the system proved reliable: the voice signal after decryption is clearly understandable, and any change in key or message content leads to the expected `InvalidToken` error, preventing the use of compromised data.

The project extension to wireless transmission of encrypted data over an RF channel using two HackRF devices confirmed that FSK modulated transmission of packetized data is feasible. However, in real conditions, loss and damage of some packets occurs, which is detected through incorrect CRC of individual frames, and the consequence is that Fernet completely refuses decryption due to failed HMAC verification and does not deliver even partial audio content.

Possible system improvements relate to the transmission layer over the RF channel and error detection in data. As a simple upgrade, multiple transmission of the same packet can be introduced, and on the receiving side, the first copy that passes integrity verification can be accepted, which would reduce the probability of packet loss, at the cost of greater channel occupation. Furthermore, instead of a simple byte sum, it is possible to use standard CRC-8 or CRC-16 with a defined polynomial and potentially introduce error correction mechanisms (FEC), so that some bit errors can be corrected even before decryption. An additional improvement is possible by implementing a mechanism in which the receiver reports which packets are missing or damaged and requests their retransmission, as well as optimizing RF parameters (baud rate, frequency spacing, gains, antennas) to reduce the number of transmission errors, which is necessary for Fernet to allow decryption of the encrypted message at all.

---

## Usage Instructions

### 1. Voice Encryption

```bash
python3 encrypt_audio.py
# Press '1' to start recording
# Press '0' to stop recording and encrypt
```

### 2. Voice Decryption

```bash
python3 decrypt_audio.py
# Press ENTER to load key from key.key or paste key manually
# Audio will play automatically after successful decryption
```

### 3. RF Transmission (Transmitter)

```bash
python3 tx_encrypted.py
# Automatically transmits encrypted_audio.bin via HackRF
```

### 4. RF Reception (Receiver)

First, start HackRF capture:
```bash
hackrf_transfer -r received_signal.iq -f 433920000 -s 2000000 -l 40 -g 20 -d [SERIAL_NUMBER]
```

Then process received data:
```bash
python3 rx_encrypted.py
# Press '1' to process IQ samples and save to .bin
# Press 'q' to quit
```

---

## Dependencies

```bash
pip install pyaudio cryptography numpy
sudo apt-get install hackrf
```

---

## Notes

- Ensure HackRF serial numbers are correctly set in the scripts
- The system requires proper RF antenna configuration for reliable transmission
- Fernet encryption requires exact key match - any key mismatch results in `InvalidToken` error
- RF transmission quality depends on distance, interference, and antenna setup
- CRC8 provides basic integrity checking but may not catch all errors in noisy environments
