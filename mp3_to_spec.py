from pathlib import Path

import pandas as pd
import soundfile as sf
import librosa
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import gc


BATCH_SIZE = 500
DATA_PATH = Path("audio")
SPECTROGRAM_OUTPUT_PATH = Path("spectrogram")

TICKS = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000])
TICK_LABELS = np.array(["31.25", "62.5", "125", "250", "500", "1k", "2k", "4k", "8k"])
SAVE_PARAMS = {"dpi": 300, "bbox_inches": "tight", "transparent": False, "pad_inches": 0}


def plot_spectrogram(signal, sample_rate, output: Path, fft_size=2048, hop_size=None, window_size=None):
    # Compute default parameters
    if not window_size:
        window_size = fft_size
    if not hop_size:
        hop_size = fft_size // 4

    # Convert audio to digital signal with fft
    stft = librosa.stft(signal, n_fft=fft_size, hop_length=hop_size, win_length=window_size, center=False)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    fig = plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='log', hop_length=hop_size, cmap = 'gray')
    plt.axis('off')

    # Plot settings
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")
    # plt.yticks(TICKS, TICK_LABELS)
    # plt.colorbar(img, format="%+2.f dBFS")

    # Save plot
    output.parent.mkdir(parents=True, exist_ok=True)
    output_path = output.with_stem(f"{output.stem}_spectrogram_win_length={window_size}_hop_length={hop_size}_n_fft={fft_size}")
    plt.savefig(output_path, **SAVE_PARAMS)
    plt.close(fig)

    # Clear memory
    del stft, spectrogram, spectrogram_db, img, fig
    gc.collect()


def process_mp3(mp3_file: Path):
    try:
        # Read mp3 files
        signal, sample_rate = sf.read(mp3_file)

        # Change 2d stereo to 1d mono
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        # This automatically places the output image in the same folder structure as the data folder.
        # Do not edit this line.
        spectrogram_file = SPECTROGRAM_OUTPUT_PATH / mp3_file.with_suffix(".png").relative_to(DATA_PATH)

        # Plot and save spectrogram
        plot_spectrogram(signal, sample_rate, spectrogram_file)
    except Exception as e:
        print(f"Error processing {mp3_file}: {e}")


def process_files():
    mp3_files = list((DATA_PATH / "test_data").rglob('*.mp3'))
    mp3_files_with_data = []

    echonest = pd.read_csv("data/echonest.csv")
    songs_data = pd.to_numeric(echonest[echonest.columns[0]], errors='coerce').dropna().astype(int).tolist()
    for mp3 in mp3_files:
        song_name = int(mp3.stem)
        # if song_name in songs_data:
        mp3_files_with_data.append(mp3)

    with Pool(int(cpu_count() / 4)) as pool:
        list(tqdm(pool.imap_unordered(process_mp3, mp3_files_with_data), total=len(mp3_files_with_data)))


if __name__ == "__main__":
    process_files()
