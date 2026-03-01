"""
===============================================================================
 STAGE 4.3: ENCRYPT LOCAL HOSPITAL WEIGHTS
===============================================================================

 Purpose:
   Each hospital has a trained local_model.pth from Stage 4.2.
   This script encrypts each hospital's model weights using CKKS-RNS
   so they can be sent to the aggregation server WITHOUT exposing
   raw parameters.

 Input:
   Model_Training/Hospital_A/local_model.pth
   Model_Training/Hospital_B/local_model.pth
   Model_Training/Hospital_C/local_model.pth

 Output:
   Encryption/encrypted/Hospital_A_weights.bin  (serialized ciphertext)
   Encryption/encrypted/Hospital_B_weights.bin
   Encryption/encrypted/Hospital_C_weights.bin
   Encryption/encrypted/weight_shapes.json      (param shapes for reconstruction)
   Encryption/encrypted/ckks_context.bin         (secret key context for decryption)

 Security:
   - CKKS-RNS with poly_modulus_degree=8192, 128-bit security
   - Server only sees ciphertexts (random-looking polynomials)
   - Without the secret key, recovering weights is computationally infeasible

===============================================================================
"""

import os
import sys
import json
import time
import torch
import tenseal as ts

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input: trained local models
HOSPITAL_DIRS = {
    "A": os.path.join(PROJECT_ROOT, "Model_Training", "Hospital_A"),
    "B": os.path.join(PROJECT_ROOT, "Model_Training", "Hospital_B"),
    "C": os.path.join(PROJECT_ROOT, "Model_Training", "Hospital_C"),
}

# Output: encrypted weights
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "encrypted")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add Model_Training to path for LSTM import
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Model_Training"))
from LSTM import MortalityLSTM


# ============================================================
# CKKS CONTEXT CREATION
# ============================================================
def create_ckks_context():
    """
    Creates a TenSEAL CKKS context with RNS representation.

    Parameters:
      poly_modulus_degree = 8192 (N)  ->  128-bit security
      coeff_mod_bit_sizes = [60, 40, 40, 60]
      global_scale = 2^40  ->  ~12 decimal digits of precision
      Max slots per ciphertext = N/2 = 4096
    """
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context


# ============================================================
# FLATTEN MODEL WEIGHTS
# ============================================================
def flatten_model_weights(state_dict):
    """
    Flattens all model parameters into a single 1D list.

    MortalityLSTM (5,921 params):
      lstm.weight_ih_l0:    (128, 8)  = 1024
      lstm.weight_hh_l0:    (128, 32) = 4096
      lstm.bias_ih_l0:      (128,)    = 128
      lstm.bias_hh_l0:      (128,)    = 128
      classifier.0.weight:  (16, 32)  = 512
      classifier.0.bias:    (16,)     = 16
      classifier.3.weight:  (1, 16)   = 16
      classifier.3.bias:    (1,)      = 1

    Returns:
      flat_weights: list of floats
      weight_shapes: dict {param_name: list(shape)} for reconstruction
    """
    flat_weights = []
    weight_shapes = {}

    for name, param in state_dict.items():
        weight_shapes[name] = list(param.shape)
        flat_weights.extend(param.cpu().numpy().flatten().tolist())

    return flat_weights, weight_shapes


# ============================================================
# ENCRYPT WEIGHTS
# ============================================================
def encrypt_weights(flat_weights, context):
    """
    Encrypts a flat weight vector into CKKS ciphertext chunks.

    Since 5,921 > 4,096 (max slots), we split into 2 chunks:
      Chunk 1: weights[0:4096]       (4,096 values)
      Chunk 2: weights[4096:5921]    (1,825 values)

    Returns:
      encrypted_chunks: list of ts.CKKSVector
    """
    max_slots = 8192 // 2  # 4096

    encrypted_chunks = []
    for i in range(0, len(flat_weights), max_slots):
        chunk = flat_weights[i:i + max_slots]
        encrypted_chunk = ts.ckks_vector(context, chunk)
        encrypted_chunks.append(encrypted_chunk)

    return encrypted_chunks


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  STAGE 4.3: ENCRYPT LOCAL HOSPITAL WEIGHTS (CKKS-RNS)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Create CKKS context (shared key for all hospitals)
    # ------------------------------------------------------------------
    print("\n  [1/3] Creating CKKS-RNS context...")
    context = create_ckks_context()
    print(f"    Poly Modulus Degree (N): 8192")
    print(f"    Coeff Moduli: [60, 40, 40, 60] bits")
    print(f"    Global Scale: 2^40 = {2**40:,}")
    print(f"    Max Slots: {8192 // 2}")
    print(f"    Security Level: 128-bit")

    # Save context (with secret key) for later decryption
    context_path = os.path.join(OUTPUT_DIR, "ckks_context.bin")
    context_bytes = context.serialize(save_secret_key=True)
    with open(context_path, "wb") as f:
        f.write(context_bytes)
    print(f"    Context saved: {context_path}")

    # ------------------------------------------------------------------
    # Step 2: Encrypt each hospital's weights
    # ------------------------------------------------------------------
    print(f"\n  [2/3] Encrypting hospital weights...")
    print("-" * 70)

    weight_shapes = None  # Same architecture, shapes are identical
    encryption_times = {}  # Track per-hospital encryption time

    for hospital_name, hospital_dir in HOSPITAL_DIRS.items():
        model_path = os.path.join(hospital_dir, "local_model.pth")

        if not os.path.exists(model_path):
            print(f"\n  Hospital {hospital_name}: SKIPPED (no local_model.pth found)")
            continue

        print(f"\n  Hospital {hospital_name}:")
        print(f"    Loading: {model_path}")

        # Load trained weights
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Flatten
        flat_weights, shapes = flatten_model_weights(state_dict)
        total_params = len(flat_weights)

        if weight_shapes is None:
            weight_shapes = shapes

        print(f"    Total Parameters: {total_params:,}")

        # Encrypt
        start_time = time.time()
        encrypted_chunks = encrypt_weights(flat_weights, context)
        enc_time = time.time() - start_time

        encryption_times[hospital_name] = enc_time

        print(f"    Ciphertext Chunks: {len(encrypted_chunks)}")
        print(f"    Encryption Time: {enc_time:.4f}s")

        # Serialize encrypted chunks and save
        out_path = os.path.join(OUTPUT_DIR, f"Hospital_{hospital_name}_weights.bin")
        serialized_chunks = []
        total_size = 0
        for chunk in encrypted_chunks:
            chunk_bytes = chunk.serialize()
            serialized_chunks.append(chunk_bytes)
            total_size += len(chunk_bytes)

        # Save as a single file: [num_chunks][len1][chunk1][len2][chunk2]...
        with open(out_path, "wb") as f:
            # Write number of chunks
            f.write(len(serialized_chunks).to_bytes(4, "little"))
            for chunk_bytes in serialized_chunks:
                # Write chunk length then chunk data
                f.write(len(chunk_bytes).to_bytes(8, "little"))
                f.write(chunk_bytes)

        size_kb = total_size / 1024
        print(f"    Ciphertext Size: {size_kb:.1f} KB")
        print(f"    Saved: {out_path}")

    # ------------------------------------------------------------------
    # Step 3: Save weight shapes (needed for reconstruction after decryption)
    # ------------------------------------------------------------------
    if weight_shapes is not None:
        shapes_path = os.path.join(OUTPUT_DIR, "weight_shapes.json")
        with open(shapes_path, "w") as f:
            json.dump(weight_shapes, f, indent=2)
        print(f"\n  [3/3] Weight shapes saved: {shapes_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ENCRYPTION COMPLETE")
    print("=" * 70)
    print(f"\n  Output Directory: {OUTPUT_DIR}")
    print(f"  Files Created:")

    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        fsize = os.path.getsize(fpath)
        if fsize > 1024 * 1024:
            print(f"    {fname:40s} {fsize / (1024*1024):.1f} MB")
        else:
            print(f"    {fname:40s} {fsize / 1024:.1f} KB")

    # ------------------------------------------------------------------
    # Write encryption report
    # ------------------------------------------------------------------
    total_enc_time = sum(encryption_times.values())
    report_path = os.path.join(OUTPUT_DIR, "encryption_report.txt")
    with open(report_path, "w") as f:
        f.write("ENCRYPTION REPORT (STAGE 4.3)\n")
        f.write("=" * 60 + "\n\n")
        f.write("CKKS-RNS Parameters:\n")
        f.write(f"  Poly Modulus Degree (N): 8192\n")
        f.write(f"  Coeff Moduli: [60, 40, 40, 60] bits\n")
        f.write(f"  Global Scale: 2^40\n")
        f.write(f"  Security Level: 128-bit\n")
        f.write(f"  Max Slots per Ciphertext: 4096\n\n")
        f.write("Per-Hospital Encryption Time:\n")
        f.write(f"  {'Hospital':<12s} {'Time (s)':<12s}\n")
        f.write(f"  {'-'*12:<12s} {'-'*12:<12s}\n")
        for name, t in encryption_times.items():
            f.write(f"  Hospital {name:<4s} {t:.4f}s\n")
        f.write(f"  {'-'*12:<12s} {'-'*12:<12s}\n")
        f.write(f"  {'Total':<12s} {total_enc_time:.4f}s\n")
        f.write(f"\nModel: MortalityLSTM (5,921 parameters)\n")
        f.write(f"Ciphertext Chunks per Hospital: 2\n")

    print(f"\n  Encryption Report saved: {report_path}")

    print(f"\n  Encryption Times:")
    for name, t in encryption_times.items():
        print(f"    Hospital {name}: {t:.4f}s")
    print(f"    Total:      {total_enc_time:.4f}s")

    print(f"\n  Next Step: Run federated aggregation (Stage 4.4)")
    print(f"    The server will homomorphically average these ciphertexts")
    print(f"    WITHOUT ever seeing the raw weights.")
    print("=" * 70)


if __name__ == "__main__":
    main()
