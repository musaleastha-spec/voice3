import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os

# Configure Page Layout
st.set_page_config(page_title="Acoustic Gender Classifier", page_icon="🎙️", layout="centered")

# Header
st.title("🎙️ Acoustic Gender Classification App")
st.write("This system extracts 50 robust acoustic features to classify the gender of voice recordings. It is highly optimized to remain invariant to regional accents, recording gear, and background noises.")

# Load Saved Model and Scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("gender_classifier_model.pkl")
    scaler = joblib.load("gender_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.success("✅ Brain & Scaler parameters loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model/scaler files. Ensure 'gender_classifier_model.pkl' and 'gender_scaler.pkl' are in the same folder as this app. Error: {e}")

# Feature extraction logic matching exactly what was used during training
def extract_50_features_from_uploaded(file_path):
    try:
        # Load audio (downsampled to 16kHz, mono, using resampy for high quality resampling)
        y, sr = librosa.load(file_path, sr=16000, mono=True, res_type='resampy')
        
        # 1. MFCCs (13 features)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        
        # 2. Chroma STFT (12 features)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        
        # 3. Mel-scaled Spectrogram (12 features)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=12), axis=1)
        
        # 4. Spectral Contrast (7 features)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6), axis=1)
        
        # 5. Spectral Centroid & Bandwidth (2 features)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # 6. Spectral Rolloff (1 feature)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # 7. Zero Crossing Rate (1 feature)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # 8. RMS Energy (1 feature)
        rms = np.mean(librosa.feature.rms(y=y))
        
        feature_vector = np.hstack([
            mfccs,
            chroma,
            mel,
            contrast,
            centroid,
            bandwidth,
            rolloff,
            zcr,
            rms
        ])
        
        return feature_vector
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (e.g., .wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Temporary file storage to allow Librosa to process
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
        
    st.write("🔄 Processing audio signals and extracting robust acoustic dimensions...")
    
    features = extract_50_features_from_uploaded(temp_path)
    
    # Clean up temp file
    os.unlink(temp_path)
    
    if features is not None:
        # Scale and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction_prob = model.predict_proba(features_scaled)[0]
        prediction_class = model.predict(features_scaled)[0]
        
        labels = {0: "Female 👩", 1: "Male 👨"}
        result = labels[prediction_class]
        confidence = prediction_prob[prediction_class] * 100
        
        st.markdown("--- ")
        st.markdown(f"### **Predicted Gender: {result}**")
        st.markdown(f"#### **Confidence level: {confidence:.2f}%**")
        
        # Progress bars for visualization
        st.write("#### Class Probabilities:")
        st.write(f"Female: {prediction_prob[0]*100:.1f}%")
        st.progress(float(prediction_prob[0]))
        st.write(f"Male: {prediction_prob[1]*100:.1f}%")
        st.progress(float(prediction_prob[1]))
