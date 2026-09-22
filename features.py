"""
Shared audio feature extraction.
IMPORTANT: train_colab.ipynb and app.py must use the EXACT same function,
otherwise the trained model and the live app will disagree on what a
feature vector means. Copy this file's logic if you change anything here.
"""

import numpy as np
import librosa


SAMPLE_RATE = 16000  # every clip is resampled to this before feature extraction


def load_audio(path_or_buffer):
    """Load any audio file and resample to a fixed rate + mono."""
    y, sr = librosa.load(path_or_buffer, sr=SAMPLE_RATE, mono=True)
    # trim leading/trailing silence so silence length doesn't leak into features
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < SAMPLE_RATE:  # pad clips shorter than 1 second
        y = np.pad(y, (0, SAMPLE_RATE - len(y)))
    return y, sr


def extract_features(y, sr=SAMPLE_RATE):
    """
    Turn a raw waveform into a fixed-length numeric feature vector.
    Returns a 1D numpy array (always the same length regardless of clip duration).
    """
    feats = []

    # --- MFCCs: the classic "timbre" descriptor, most TTS/vocoder artifacts show up here
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats += list(np.mean(mfcc, axis=1))
    feats += list(np.std(mfcc, axis=1))

    # --- Delta MFCC: how fast timbre changes frame-to-frame (cloned voices are often "too smooth")
    delta = librosa.feature.delta(mfcc)
    feats += list(np.mean(delta, axis=1))
    feats += list(np.std(delta, axis=1))

    # --- Spectral shape descriptors
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats += [np.mean(centroid), np.std(centroid)]

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats += [np.mean(bandwidth), np.std(bandwidth)]

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats += [np.mean(rolloff), np.std(rolloff)]

    flatness = librosa.feature.spectral_flatness(y=y)
    feats += [np.mean(flatness), np.std(flatness)]

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats += list(np.mean(contrast, axis=1))

    # --- Time-domain descriptors
    zcr = librosa.feature.zero_crossing_rate(y)
    feats += [np.mean(zcr), np.std(zcr)]

    rms = librosa.feature.rms(y=y)
    feats += [np.mean(rms), np.std(rms)]

    # --- Pitch / prosody (jitter-like): natural voices vary more than most TTS
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    if len(f0_voiced) > 1:
        feats += [np.nanmean(f0_voiced), np.nanstd(f0_voiced)]
    else:
        feats += [0.0, 0.0]

    # --- Chroma (harmonic content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats += list(np.mean(chroma, axis=1))

    return np.array(feats, dtype=np.float32)


FEATURE_VECTOR_LENGTH = None  # computed lazily below for documentation purposes only


def waveform_preview(y, n_points=300):
    """Downsample the waveform to ~n_points values, for drawing in the browser."""
    if len(y) <= n_points:
        return y.tolist()
    idx = np.linspace(0, len(y) - 1, n_points).astype(int)
    return y[idx].tolist()
