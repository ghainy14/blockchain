# app.py

import os
import pandas as pd
import numpy as np
import json
import io
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time, hashlib

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
    
    def add_transaction(self, transaction):
        if self.verify_transaction(transaction):
            self.unconfirmed_transactions.append(transaction)
            return True
        return False

    def verify_transaction(self, transaction):
        expected_signature = hashlib.sha256(
            f"{transaction.device_id}{transaction.model_update}{transaction.timestamp}".encode()
        ).hexdigest()
        return transaction.signature == expected_signature
    
    def proof_of_work(self, block):
        block.nonce = 0
        computed_hash = block.compute_hash()
        while not computed_hash.startswith('0' * self.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()
        return computed_hash
    
    def add_block(self, block, proof):
        if self.last_block().hash != block.previous_hash:
            return False
        if not proof.startswith('0' * self.difficulty) or proof != block.compute_hash():
            return False
        self.chain.append(block)
        return True
    
    def mine(self):
        if not self.unconfirmed_transactions:
            return None
        new_block = Block(
            index=self.last_block().index + 1,
            transactions=self.unconfirmed_transactions,
            previous_hash=self.last_block().hash
        )
        proof = self.proof_of_work(new_block)
        self.add_block(new_block, proof)
        self.unconfirmed_transactions = []
        return new_block

# -------------------------------
# Model Definition
# -------------------------------
def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# Training and Blockchain Storage
# -------------------------------
def train_and_store(dataset_list, num_epochs=3, batch_size=32):
    blockchain = Blockchain(difficulty=2)
    metrics = {"Device": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-Score": [], "AUC": []}

    for device_id, dataset in enumerate(dataset_list, start=1):
        df = pd.read_csv(dataset)
        df_numeric = df.select_dtypes(include=[np.number])

        X = df_numeric.iloc[:, :-1].values.astype(np.float32)
        y = df_numeric.iloc[:, -1].values.astype(np.float32)
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

        input_dim = X.shape[1]
        model = create_model(input_dim)
        model.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=0)

        y_pred = (model.predict(X) > 0.5).astype(int)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=1)
        recall = recall_score(y, y_pred, zero_division=1)
        f1 = f1_score(y, y_pred, zero_division=1)
        auc = roc_auc_score(y, y_pred)

        metrics["Device"].append(f"Device {device_id}")
        metrics["Accuracy"].append(accuracy)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["F1-Score"].append(f1)
        metrics["AUC"].append(auc)

        # Store model update in blockchain
        weights = model.get_weights()
        transaction = Transaction(device_id=f"Device_{device_id}", model_update={"weights": [w.tolist() for w in weights]})
        blockchain.add_transaction(transaction)
        blockchain.mine()

    return metrics, blockchain

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Federated Learning + Blockchain", layout="wide")
st.title("🔗 Federated Learning with Blockchain + Deep Learning Evaluation")

uploaded_files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Train & Evaluate"):
        with st.spinner("Training models... please wait"):
            metrics, blockchain = train_and_store(uploaded_files)

        st.success("✅ Training completed!")

        # Show metrics as table
        df_metrics = pd.DataFrame(metrics).set_index("Device")
        st.subheader("📊 Model Performance Summary")
        st.dataframe(df_metrics.style.highlight_max(axis=0))

        # 📥 Download Metrics as CSV
        csv_buffer = io.StringIO()
        df_metrics.to_csv(csv_buffer)
        st.download_button(
            label="⬇️ Download Metrics (CSV)",
            data=csv_buffer.getvalue(),
            file_name="model_metrics.csv",
            mime="text/csv"
        )

        # Plot metrics
        st.subheader("📈 Performance Visualization")
        fig, ax = plt.subplots(figsize=(10, 5))
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]:
            ax.plot(metrics["Device"], metrics[metric], marker="o", label=metric)
        ax.set_xlabel("Devices")
        ax.set_ylabel("Scores")
        ax.set_title("Model Performance Across IIoT Devices")
        ax.legend()
        st.pyplot(fig)

        # Blockchain details
        st.subheader("⛓️ Blockchain Ledger")
        blockchain_data = []
        for block in blockchain.chain:
            block_dict = {
                "index": block.index,
                "hash": block.hash,
                "previous_hash": block.previous_hash,
                "transactions": [tx.to_dict() for tx in block.transactions],
                "timestamp": block.timestamp
            }
            blockchain_data.append(block_dict)
            st.json(block_dict)

        # 📥 Download Blockchain as JSON
        st.download_button(
            label="⬇️ Download Blockchain Ledger (JSON)",
            data=json.dumps(blockchain_data, indent=4),
            file_name="blockchain_ledger.json",
            mime="application/json"
        )
