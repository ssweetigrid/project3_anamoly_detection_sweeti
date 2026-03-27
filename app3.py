import streamlit as st
import numpy as np
import pickle
import torch
import torch.nn as nn

# -----------------------------
# MODEL
# -----------------------------
class TransformerAE(nn.Module):
    def __init__(self, input_dim=2048, d_model=256):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.decoder(x)

# -----------------------------
# LOAD MODEL
# -----------------------------
device = "cpu"

@st.cache_resource
def load_model():
    model = TransformerAE().to(device)
    model.load_state_dict(torch.load(
        "/Users/ssweeti/Desktop/anomaly_detection_proejct3/transformer_model/model.pth",
        map_location=device
    ))
    model.eval()
    return model

model = load_model()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    Z = np.load("Z.npy")
    features_seq = np.load("test_seq.npy")

    with open("concept_vectors.pkl", "rb") as f:
        concept_vectors = pickle.load(f)

    return Z, features_seq, concept_vectors

Z, features_seq, concept_vectors = load_data()

# -----------------------------
# UI
# -----------------------------
st.title("🚀 Concept Steering Demo (Fast Version)")

alpha_driving = st.slider("Driving sensitivity", -0.5, 0.5, 0.0)
alpha_lane = st.slider("Lane sensitivity", -0.5, 0.5, 0.0)
alpha_waiting = st.slider("Waiting sensitivity", -0.5, 0.5, 0.0)

# sample size control
N = st.slider("Number of samples (speed control)", 500, 5000, 2000)

# -----------------------------
# USE SMALL SUBSET
# -----------------------------
Z_small = Z[:N]
features_small = features_seq[:N]

# -----------------------------
# STEERING
# -----------------------------
Z_steered = Z_small.copy()

if "driving" in concept_vectors:
    Z_steered += alpha_driving * concept_vectors["driving"]

if "lane" in concept_vectors:
    Z_steered += alpha_lane * concept_vectors["lane"]

if "waiting" in concept_vectors:
    Z_steered += alpha_waiting * concept_vectors["waiting"]

# -----------------------------
# RECONSTRUCTION SCORE (FAST)
# -----------------------------
Z_seq = np.repeat(Z_steered[:, None, :], 8, axis=1)

with torch.no_grad():
    z_tensor = torch.tensor(Z_seq, dtype=torch.float32).to(device)
    recon = model.decoder(z_tensor)

    input_tensor = torch.tensor(features_small, dtype=torch.float32).to(device)

    loss = ((recon - input_tensor) ** 2).mean(dim=(1,2))

score = loss.cpu().numpy()

# -----------------------------
# DISPLAY
# -----------------------------
st.subheader("📊 Anomaly Scores")

st.write("Mean score:", float(np.mean(score)))
st.write("Max score:", float(np.max(score)))

st.write("Sample scores:", score[:10])

# -----------------------------
# VISUALIZATION
# -----------------------------
st.subheader("📈 Score Distribution")

st.line_chart(score[:200])