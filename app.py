import os
import logging
import uuid
import traceback
from datetime import datetime
from typing import Optional, Tuple

import librosa
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import subprocess
from dataclasses import dataclass
from enum import Enum

# ----------------------
# Configuration
# ----------------------
class Config:
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
    UPLOAD_FOLDER = 'uploads'
    CONVERTED_FOLDER = 'converted'
    ALLOWED_EXTENSIONS = {'aac', 'wav', 'mp3', 'ogg'}

# ----------------------
# Enums and Data Classes
# ----------------------
class VoiceType(Enum):
    SOPRANO = "Soprano"
    MEZZO_SOPRANO = "Mezzo-soprano"
    CONTRALTO = "Kontralto"
    COUNTER_TENOR = "Tiz Erkek (Countertenor)"
    TENOR = "Tenor"
    BARITONE = "Bariton"
    BASS = "Bas"
    UNKNOWN = "Belirsiz"

@dataclass
class AnalysisResult:
    voice_type: VoiceType
    average_pitch: float
    processing_time: str
    message: Optional[str] = None

# ----------------------
# Flask App Setup
# ----------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# ----------------------
# Logging Setup
# ----------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('api.log'),
            logging.StreamHandler()
        ]
    )

# ----------------------
# Helper Functions
# ----------------------
def create_folders():
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.CONVERTED_FOLDER, exist_ok=True)
    logging.info("Upload ve converted klasörleri kontrol edildi.")

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename: str) -> Tuple[str, str]:
    unique_id = uuid.uuid4().hex
    safe_filename = secure_filename(f"{unique_id}_{original_filename}")
    input_path = os.path.join(Config.UPLOAD_FOLDER, safe_filename)
    output_path = os.path.join(Config.CONVERTED_FOLDER, f"{os.path.splitext(safe_filename)[0]}.wav")
    return input_path, output_path

def clean_up_files(*file_paths):
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                logging.info(f"Silindi: {path}")
        except Exception as e:
            logging.warning(f"Dosya silme hatası ({path}): {str(e)}")

# ----------------------
# Audio Processing
# ----------------------
def convert_to_wav(input_path: str, output_path: str) -> None:
    try:
        subprocess.run(
            ['ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ar', '44100', output_path, '-y'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"✅ Dönüştürüldü: {input_path} → {output_path}")
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg hatası: {e.stderr}"
        logging.error(error_msg)
        raise Exception(error_msg)

def analyze_pitch(y: np.ndarray, sr: int) -> Tuple[float, list]:
    if len(y) / sr < 0.5:
        raise ValueError("Ses dosyası çok kısa (en az 0.5 saniye)")

    pitch = librosa.yin(y, fmin=50, fmax=1000, sr=sr)
    pitch = pitch[np.isfinite(pitch)]
    pitch = pitch[pitch > 0]

    if pitch.size == 0:
        raise ValueError("Pitch bulunamadı")

    return float(np.median(pitch)), pitch.tolist()

def classify_voice(avg_pitch: float, gender: str) -> VoiceType:
    gender = gender.lower()

    if gender == 'female':
        if 450 <= avg_pitch <= 1100:
            return VoiceType.SOPRANO
        elif 350 <= avg_pitch < 450:
            return VoiceType.MEZZO_SOPRANO
        elif 165 <= avg_pitch < 350:
            return VoiceType.CONTRALTO
        else:
            return VoiceType.UNKNOWN

    elif gender == 'male':
        if 450 <= avg_pitch <= 700:
            return VoiceType.COUNTER_TENOR
        elif 165 <= avg_pitch < 255:
            return VoiceType.TENOR
        elif 110 <= avg_pitch < 165:
            return VoiceType.BARITONE
        elif 80 <= avg_pitch < 110:
            return VoiceType.BASS
        else:
            return VoiceType.UNKNOWN

    return VoiceType.UNKNOWN

# ----------------------
# API Endpoints
# ----------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = datetime.now()
    input_path, output_path = None, None

    try:
        print("🎙️ /analyze isteği alındı.")
        if 'file' not in request.files:
            raise ValueError("Dosya yüklenmedi (file alanı eksik)")
        if 'gender' not in request.form:
            raise ValueError("Cinsiyet bilgisi eksik")
        
        gender = request.form['gender']
        file = request.files['file']

        if file.filename == '':
            raise ValueError("Dosya seçilmedi")
        if not allowed_file(file.filename):
            raise ValueError("Desteklenmeyen dosya formatı")

        input_path, output_path = generate_unique_filename(file.filename)
        logging.info(f"Yüklenen dosya: {input_path}")
        file.save(input_path)
        convert_to_wav(input_path, output_path)

        y, sr = librosa.load(output_path, sr=None)
        avg_pitch, pitch_series = analyze_pitch(y, sr)
        voice_type = classify_voice(avg_pitch, gender)

        result = AnalysisResult(
            voice_type=voice_type,
            average_pitch=round(avg_pitch, 2),
            processing_time=str(datetime.now() - start_time)
        )

        logging.info(f"Analiz sonucu: {result}")

        return jsonify({
            'status': 'success',
            'voice_type': result.voice_type.value,
            'average_pitch': result.average_pitch,
            'pitch_series': pitch_series,
            'processing_time': result.processing_time
        })

    except ValueError as e:
        logging.error(f"❌ Validation hatası: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except RequestEntityTooLarge:
        error_msg = "Dosya boyutu 10MB sınırını aşıyor"
        logging.error(error_msg)
        return jsonify({'status': 'error', 'message': error_msg}), 413
    except Exception as e:
        logging.error(f"❌ İstisna oluştu: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        clean_up_files(input_path, output_path)

@app.route('/health', methods=['GET'])
def health_check():
    ffmpeg_available = subprocess.run(['ffmpeg', '-version'], capture_output=True).returncode == 0
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'ffmpeg': 'available' if ffmpeg_available else 'unavailable',
            'storage': {
                'upload_folder': os.path.isdir(Config.UPLOAD_FOLDER),
                'converted_folder': os.path.isdir(Config.CONVERTED_FOLDER)
            }
        }
    })

@app.route('/')
def home():
    return jsonify({
        'message': 'VocalCoach API çalışıyor!',
        'endpoints': {
            'analyze': '/analyze (POST)',
            'health_check': '/health (GET)'
        }
    })

# ----------------------
# Initialization
# ----------------------
setup_logging()
create_folders()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
