import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    mp3_files = list(Path("audio/fma_small").rglob("*.mp3"))

    echonest = pd.read_csv("data/echonest.csv")
    songs_data = pd.to_numeric(echonest[echonest.columns[0]], errors='coerce').dropna().astype(int).tolist()

    count = 0

    for mp3 in mp3_files:
        song_name = int(mp3.stem)
        if song_name in songs_data:
            count += 1
    print(count)