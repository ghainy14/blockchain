# app.py

# --- Optional protobuf safety for some TF deployments (put BEFORE tf import) ---
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import io
import json
import time
import hashlib
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# -------------------------------
# Blockchain Implementation
# -------------------------------
class Transaction:
    def __init__(self, device_id, model_update, timestamp=None, signature=None):
        self.device_id = device_id
        self.model_update = model_update
        self.timestamp = timestamp or time.time()
        self.signature = signature or self.sign_transaction()

    def sign_transaction(self):
        tx_str = f"{self.device_id}{self.model_update}{self.timestamp}"
        return hashlib.sha256(tx_str.encode()).hexdigest()

    def to_dict(self):
        return {
            "device_id": self.device_id,
            "model_update": self.model_update,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }

class Block:
    def __init__(self, index, transactions, previous_hash, timestamp=None, nonce=0):
        self.index = index
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.timestamp = timestamp or time.time()
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_content = {
            "index": self.index,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }
        block_string = json.dumps(block_content, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self, difficulty=2):
        self.unconfirmed_transactions = []
        self.chain = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, [], "0", time.time())
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)

    def last_block(self):
        return self.chain[-1]

    def add_transaction(self, transaction: Transaction):
        if self.verify_transaction(transaction):
            self.unconfirmed_transactions.append(transaction)
            return True
        return False

    def verify_transaction(self, transaction: Transaction):
        expected_signature = hashlib.sha256(
            f"{transaction.device_id}{transaction.model_update}{transaction.timestamp}".encode()
        ).hexdigest()
        return transaction.signature == expected_signature

    def proof_of_work(self, block: Block):
        block.nonce = 0
        computed_hash = block.compute_hash()
        while not computed_hash.startswith("0" * self.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()
        return computed_hash

    def add_block(self, block: Block, proof: str):
        if self.last_block().hash != block.previous_hash:
            return False
        if not proof.startswith("0" * self.difficulty) or proof != block.compute_hash():
            return False
        self.chain.append(block)
        return True

    def mine(self):
        if not self.unconfirmed_transactions:
            return None
        new_block = Block(
            index=self.last_block().index + 1,
            transactions=self.unconfirmed_transactions,
            previous_hash=self.last_block().hash,
        )
        proof = self.proof_of_work(new_block)
        self.add_block(new_block, proof)
        self.unconfirmed_transactions = []
        return new_block

# -------------------------------
# Deep Learning Model Definition
# -------------------------------
def create_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),  # Binary classification
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

# -------------------------------
# Training and Blockchain Storage
# -------------------------------
def train_and_store(
    dataset_list: List[io.BytesIO],
    num_epochs: int = 3,
    batch_size: int = 32,
    difficulty: int = 2,
    save_path: str = "models",
) -> Tuple[dict, Blockchain, List[str]]:
    """
    Trains a separate model per uploaded CSV (each is a device),
    stores model updates as blockchain transactions, and returns metrics,
    the blockchain, and paths to saved .h5 models.
    """
    os.makedirs(save_path, exist_ok=True)
    blockchain = Blockchain(difficulty=difficulty)

    metrics = {
        "Device": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
        "AUC": [],
    }
    saved_models: List[str] = []

    for device_id, dataset in enumerate(dataset_list, start=1):
        # dataset is a Streamlit UploadedFile or file-like object; pandas can read it directly
        df = pd.read_csv(dataset, low_memory=False)
        df_numeric = df.select_dtypes(include=[np.number])

        # sanity checks
        if df_numeric.shape[1] < 2:
            # need at least 1 feature + 1 target column
            st.warning(
                f"Device {device_id}: Not enough numeric columns. "
                "Ensure the last numeric column is the binary target and others are features."
            )
            continue

        # Features = all numeric columns except last; Target = last numeric column
        X = df_numeric.iloc[:, :-1].values.astype(np.float32)
        y = df_numeric.iloc[:, -1].values.astype(np.float32)

        # Normalize features
        denom = (X.max(axis=0) - X.min(axis=0))
        denom[denom == 0] = 1.0  # avoid div by zero for constant columns
        X = (X - X.min(axis=0)) / (denom + 1e-8)

        input_dim = X.shape[1]
        model = create_model(input_dim)
        model.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=0)

        # Save model per device
        model_filename = os.path.join(save_path, f"device_{device_id}_model.h5")
        model.save(model_filename)
        saved_models.append(model_filename)

        # Predictions
        y_proba = model.predict(X, verbose=0).ravel()
        y_pred = (y_proba > 0.5).astype(int)

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=1)
        recall = recall_score(y, y_pred, zero_division=1)
        f1 = f1_score(y, y_pred, zero_division=1)
        try:
            auc = roc_auc_score(y, y_proba)  # use probabilities for better AUC
        except ValueError:
            # fallback if y has a single class
            auc = float("nan")

        metrics["Device"].append(f"Device {device_id}")
        metrics["Accuracy"].append(accuracy)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["F1-Score"].append(f1)
        metrics["AUC"].append(auc)

        # Store weights on blockchain as a transaction (converted to lists for JSON)
        weights = model.get_weights()
        tx = Transaction(
            device_id=f"Device_{device_id}",
            model_update={"weights": [w.tolist() for w in weights]},
        )
        blockchain.add_transaction(tx)
        blockchain.mine()

    return metrics, blockchain, saved_models

def plot_metrics(metrics: dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    devices = metrics["Device"]
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]:
        # replace NaNs (e.g., AUC) with np.nan-safe values for plotting
        yvals = pd.Series(metrics[metric], index=devices).astype(float)
        ax.plot(devices, yvals, marker="o", label=metric)
    ax.set_xlabel("Devices")
    ax.set_ylabel("Scores")
    ax.set_title("Model Performance Across IIoT Devices")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def zip_models(model_paths: List[str]) -> bytes:
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in model_paths:
            arcname = os.path.basename(p)
            zf.write(p, arcname=arcname)
    mem_zip.seek(0)
    return mem_zip.read()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Federated Learning + Blockchain", layout="wide")

st.title("🔗 Federated Learning with Blockchain + Deep Learning Evaluation")
st.caption(
    "Upload one or more CSV files (each CSV represents a device). "
    "The app trains a separate model per device, records model updates on a local blockchain, "
    "and reports metrics."
)

with st.sidebar:
    st.header("⚙️ Training Settings")
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=3, step=1)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=1024, value=32, step=1)
    difficulty = st.slider("Blockchain PoW Difficulty", min_value=1, max_value=5, value=2, step=1)
    models_dir = st.text_input("Models Save Path", value="models")

uploaded_files = st.file_uploader(
    "Upload one or more CSV files", type=["csv"], accept_multiple_files=True
)

if uploaded_files:
    if st.button("🚀 Train & Evaluate"):
        with st.spinner("Training models and mining blocks..."):
            metrics, blockchain, saved_models = train_and_store(
                uploaded_files,
                num_epochs=int(epochs),
                batch_size=int(batch_size),
                difficulty=int(difficulty),
                save_path=models_dir,
            )

        st.success("✅ Training completed!")

        # --- Metrics Table ---
        if metrics["Device"]:
            df_metrics = pd.DataFrame(metrics).set_index("Device")
            st.subheader("📊 Model Performance Summary")
            st.dataframe(df_metrics.style.format("{:.4f}").highlight_max(axis=0), use_container_width=True)

            # Download metrics CSV
            csv_buf = io.StringIO()
            df_metrics.to_csv(csv_buf)
            st.download_button(
                label="⬇️ Download Metrics (CSV)",
                data=csv_buf.getvalue(),
                file_name="model_metrics.csv",
                mime="text/csv",
            )

            # --- Plot ---
            st.subheader("📈 Performance Visualization")
            fig = plot_metrics(metrics)
            st.pyplot(fig)

        else:
            st.warning("No valid devices were trained. Please check your CSVs (need numeric features and a numeric target in the last numeric column).")

        # --- Blockchain Ledger ---
        st.subheader("⛓️ Blockchain Ledger")
        blockchain_data = []
        for block in blockchain.chain:
            block_dict = {
                "index": block.index,
                "hash": block.hash,
                "previous_hash": block.previous_hash,
                "transactions": [tx.to_dict() for tx in block.transactions],
                "timestamp": block.timestamp,
            }
            blockchain_data.append(block_dict)
            with st.expander(f"Block #{block.index}"):
                st.json(block_dict)

        st.download_button(
            label="⬇️ Download Blockchain Ledger (JSON)",
            data=json.dumps(blockchain_data, indent=2),
            file_name="blockchain_ledger.json",
            mime="application/json",
        )

        # --- Download trained models as a ZIP ---
        if saved_models:
            try:
                models_zip_bytes = zip_models(saved_models)
                st.download_button(
                    label="⬇️ Download All Trained Models (.zip)",
                    data=models_zip_bytes,
                    file_name="trained_models.zip",
                    mime="application/zip",
                )
            except Exception as e:
                st.warning(f"Could not prepare ZIP of models: {e}")

else:
    st.info("👆 Upload at least one CSV to begin. Each CSV should have numeric features and the **last numeric column** as the binary target (0/1).")
