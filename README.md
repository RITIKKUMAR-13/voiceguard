# Voice Cloning Detector — SIH26104

Do hisso mein kaam hai: **training** (Google Colab mein) aur **app** (Antigravity /
apne laptop pe).

---

## Part 1 — Colab mein model train karo

1. Apne 10+ real aur 10+ fake clips ko is structure mein rakho aur ek ZIP banao:
   ```
   clips.zip
   ├── real/
   │   ├── real1.wav
   │   ├── real2.wav
   └── fake/
       ├── fake1.wav
       ├── fake2.wav
   ```
   Kam clips hain toh pehle ffmpeg se 3-second chunks bana lo (notebook ke Cell 6 mein command diya hai).

2. [Google Colab](https://colab.research.google.com) kholo, `train_colab.ipynb` upload karo
   (File → Upload notebook).

3. Runtime → Change runtime type → **T4 GPU** chuno.

4. Cells ko upar se neeche, ek ek karke chalao (Shift+Enter). Cell 2 mein apni `clips.zip`
   upload karni hai.

5. Cell 5 ka output dekhna — yeh tumhara test accuracy hai. Cell 6 warning degi agar data kam hai.

6. Last cell (Cell 7) do files download karega: **model.pkl** aur **scaler.pkl**.
   Inhe is project ke `model/` folder mein daal do (agar folder nahi hai, bana lo).

---

## Part 2 — App chalao (Antigravity ya apne laptop pe)

1. Antigravity kholo, is poore `voice_clone_detector` folder ko project ke roop mein kholo.

2. Terminal mein:
   ```
   pip install -r requirements.txt
   ```

3. Confirm karo ki `model/model.pkl` aur `model/scaler.pkl` dono files hain
   (Part 1 se download ki hui).

4. App chalao:
   ```
   python app.py
   ```

5. Browser mein kholo: **http://127.0.0.1:5000**

6. Audio upload karo — result screen pe REAL/FAKE, confidence %, aur waveform dikhega.

---

## Demo ke liye zaroori baatein

- **Demo se pehle**: apni saari real/fake clips ek baar app pe chala ke dekh lo, taaki live
  demo mein surprise na ho.
- **Judges ko honestly bolo**: "Yeh prototype hamare training clips ke set pe X% accurate hai;
  bade dataset (ASVspoof) pe test karna future scope hai." Fake "94% accurate" claim mat karna.
- Agar `model/model.pkl` missing hai, app chalega par upar red warning dikhayega aur
  Analyze button disabled rahega — pehle Part 1 poora karo.

## Files

| File | Kaam |
|---|---|
| `train_colab.ipynb` | Colab notebook — model train karta hai |
| `features.py` | Audio se feature nikalne ka common code (app is use karta hai) |
| `app.py` | Flask backend — upload leta hai, predict karta hai |
| `templates/index.html` | Web page — upload UI, result card, waveform |
| `model/model.pkl`, `model/scaler.pkl` | Colab se download ki hui trained files (khud daalni hain) |
