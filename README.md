---
language:
- ar
base_model:
- SparkAudio/Spark-TTS-0.5B
tags:
- speech
- arabic
- spark
- tts
- text-to-speech
license: fair-noncommercial-research-license
---

# 🇹🇳  Tunisian_TTS

This repository provides a production-ready **Tunisian Text-to-Speech (TTS)** model based on the Spark-TTS architecture. Fine-tuned on 300 hours of high-quality Modern Standard Arabic (MSA) audio, it delivers natural prosody, full diacritization support, and zero-shot voice cloning.

## 🚀 Features

* **High-Fidelity Synthesis**: Optimized for Modern Standard Arabic (MSA) at a 24kHz sample rate.
* **Zero-Shot Voice Cloning**: Clone any voice instantly using a short reference audio clip.
* **Long-Form Support**: Integrated text chunking and crossfading for synthesizing long articles.
* **Production Ready**: Advanced audio post-processing (normalization, silence removal).
* **Deployment Ready**: Compatible with Hugging Face Spaces and easily containerized.

## 📂 Project Structure

Arabic_Spark_TTS/
├── app.py                # Main application/API logic
├── requirements.txt      # Python dependencies
├── model/                # Local model weights and config
├── testing.py            # Script for local inference testing
└── samples/              # Audio reference samples
## 🛠️ Local Setup

### 1. Install Dependencies

```bash
pip install transformers soundfile torch accelerate
2. Run Locally (Python)Pythonfrom transformers import AutoProcessor, AutoModel
import soundfile as sf
import torch

model_id = "IbrahimSalah/Arabic-TTS-Spark"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model & Processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval().to(device)

# Prepare inputs (Requires Tashkeel)
inputs = processor(
    text="إِنَّ الْعِلْمَ نُورٌ يُقْذَفُ فِي الْقَلْبِ",
    prompt_speech_path="reference.wav",
    prompt_text="النص الخاص بالمقطع الصوتي المرجعي",
    return_tensors="pt"
).to(device)

# Generate
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=8000, temperature=0.8)

# Save Output
output = processor.decode(generated_ids=output_ids)
sf.write("output.wav", output["audio"], output["sampling_rate"])
```

## ⚠️ Input Requirements 

##### CRITICAL:
The model is trained exclusively on fully diacritized text (Tashkeel). Inputting plain Arabic text without diacritics will result in poor pronunciation and robotic prosody.

✅ Correct: إِنَّ الدَّوْلَةَ لَهَا أَعْمَارٌ طَبِيعِيَّةٌ

❌ Incorrect: ان الدولة لها اعمار طبيعية

☁️ Deployment & Demos

Interactive Access
Hugging Face Space: Arabic Spark TTS Space
Colab Notebook: Try on Google Colab

| Parameter           | Default | Description                                      |
|--------------------|---------|--------------------------------------------------|
| temperature         | 0.8     | Controls randomness (higher = more expressive) |
| max_chunk_length    | 300     | Max characters per processing block             |
| crossfade_duration  | 0.08    | Seconds of overlap between text chunks          |
| top_p               | 0.95    | Nucleus sampling threshold                       |

🎧 Sample Output

Input: "إِنَّ الدَّوْلَةَ لَهَا أَعْمَارٌ طَبِيعِيَّةٌ كَمَا لِلْأَشْخَاصِ..."

Generated Audio:
<audio controls src="https://cdn-uploads.huggingface.co/production/uploads/645098004f731658826cfe57/FCGgeIu1F89rvNI55aVIx.wav"></audio>

Reference Audio:
<audio controls src="https://cdn-uploads.huggingface.co/production/uploads/645098004f731658826cfe57/cA9Z77_P0Rm2-hu1eosOC.wav"></audio>
