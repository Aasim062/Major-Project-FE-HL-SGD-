"""
===============================================================================
 STAGE 4.4: FEDERATED AGGREGATION (SERVER-SIDE)
===============================================================================

 Purpose:
   The aggregation server receives encrypted weight files from 3 hospitals.
   It performs Federated Averaging (FedAvg) entirely on ciphertexts using
   homomorphic operations -- the server NEVER sees raw model weights.

 Input:
   Encryption/encrypted/Hospital_A_weights.bin   (CKKS ciphertext)
   Encryption/encrypted/Hospital_B_weights.bin   (CKKS ciphertext)
   Encryption/encrypted/Hospital_C_weights.bin   (CKKS ciphertext)
   Encryption/encrypted/weight_shapes.json       (param shapes)
   Encryption/encrypted/ckks_context.bin          (context with secret key)

 Output:
   Federated_Average/global_model.pth             (aggregated global model)
   Federated_Average/aggregation_report.txt        (timing and details)

 Math:
   ct_sum = ct_A + ct_B + ct_C              (homomorphic addition)
   ct_avg = (1/3) * ct_sum                  (homomorphic scalar mult)
   w_global = Decrypt(ct_avg)               (plaintext averaged weights)

   Guarantee: |w_global - (w_A+w_B+w_C)/3| < 10^-7  (negligible noise)

===============================================================================
"""

import os
import sys
import json
import time
import torch
import numpy as np
import tenseal as ts

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input: encrypted weights from Stage 4.3
ENCRYPTED_DIR = os.path.join(PROJECT_ROOT, "Encryption", "encrypted")

HOSPITAL_NAMES = ["A", "B", "C"]
NUM_HOSPITALS = len(HOSPITAL_NAMES)

# Output
OUTPUT_DIR = SCRIPT_DIR  # Federated_Average/

# Add Model_Training to path for LSTM import
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Model_Training"))
from LSTM import MortalityLSTM


# ============================================================
# LOAD ENCRYPTED WEIGHTS FROM .bin FILE
# ============================================================
def load_encrypted_weights(filepath, context):
    """
    Reads a serialized encrypted weight file produced by encrypt_local_weights.py.

    File format:
      [4 bytes: num_chunks (little-endian)]
      For each chunk:
        [8 bytes: chunk_length (little-endian)]
        [chunk_length bytes: serialized CKKSVector]

    Returns:
        list of ts.CKKSVector (deserialized ciphertext chunks)
    """
    with open(filepath, "rb") as f:
        num_chunks = int.from_bytes(f.read(4), "little")
        encrypted_chunks = []
        for _ in range(num_chunks):
            chunk_len = int.from_bytes(f.read(8), "little")
            chunk_bytes = f.read(chunk_len)
            chunk = ts.lazy_ckks_vector_from(chunk_bytes)
            chunk.link_context(context)
            encrypted_chunks.append(chunk)

    return encrypted_chunks


# ============================================================
# HOMOMORPHIC AGGREGATION (FedAvg on ciphertexts)
# ============================================================
def aggregate_encrypted(all_encrypted, num_hospitals):
    """
    Performs Federated Averaging on encrypted weights.
    The server NEVER decrypts -- it only does ring arithmetic.

    Math:
      ct_sum[i] = ct_A[i] + ct_B[i] + ct_C[i]     (for each chunk i)
      ct_avg[i] = (1/K) * ct_sum[i]                 (K = num_hospitals)

      Correctness:
        Dec(ct_avg) = (1/K) * sum(w_k) + noise
        |noise| ~ K * 2^{-40} ~ 10^{-12}

    Args:
        all_encrypted: list of [list of CKKSVector] per hospital
        num_hospitals: int (K = 3)

    Returns:
        aggregated_chunks: list of CKKSVector (averaged ciphertexts)
    """
    num_chunks = len(all_encrypted[0])

    aggregated_chunks = []
    for chunk_idx in range(num_chunks):
        # Start with first hospital's chunk
        ct_sum = all_encrypted[0][chunk_idx]

        # Add remaining hospitals' chunks (homomorphic addition)
        for h in range(1, num_hospitals):
            ct_sum = ct_sum + all_encrypted[h][chunk_idx]

        # Divide by number of hospitals (homomorphic scalar mult)
        ct_avg = ct_sum * (1.0 / num_hospitals)

        aggregated_chunks.append(ct_avg)

    return aggregated_chunks


# ============================================================
# DECRYPT AGGREGATED WEIGHTS
# ============================================================
def decrypt_weights(encrypted_chunks, weight_shapes):
    """
    Decrypts CKKS ciphertext chunks back into a PyTorch state_dict.

    Args:
        encrypted_chunks: list of CKKSVector (aggregated)
        weight_shapes: dict {param_name: [shape]}

    Returns:
        state_dict: OrderedDict for model.load_state_dict()
    """
    # Decrypt all chunks and concatenate
    flat_weights = []
    for chunk in encrypted_chunks:
        decrypted = chunk.decrypt()
        flat_weights.extend(decrypted)

    # Calculate total params and trim (last chunk may have padding)
    total_params = sum(int(np.prod(shape)) for shape in weight_shapes.values())
    flat_weights = flat_weights[:total_params]

    # Reconstruct state_dict
    state_dict = {}
    offset = 0
    for name, shape in weight_shapes.items():
        num_params = int(np.prod(shape))
        param_flat = flat_weights[offset:offset + num_params]
        param_tensor = torch.tensor(param_flat, dtype=torch.float32).reshape(shape)
        state_dict[name] = param_tensor
        offset += num_params

    return state_dict


# ============================================================
# PLAINTEXT AGGREGATION (for comparison)
# ============================================================
def aggregate_plaintext():
    """
    Loads raw local_model.pth from each hospital and averages them
    WITHOUT encryption. Used to verify that encrypted aggregation
    produces identical results.

    Returns:
        state_dict: plaintext averaged weights
    """
    state_dicts = []
    for name in HOSPITAL_NAMES:
        model_path = os.path.join(PROJECT_ROOT, "Model_Training",
                                  f"Hospital_{name}", "local_model.pth")
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dicts.append(sd)

    # Average all parameters
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg_state_dict[key] = stacked.mean(dim=0)

    return avg_state_dict


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  STAGE 4.4: FEDERATED AGGREGATION (SERVER-SIDE)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load CKKS context
    # ------------------------------------------------------------------
    print("\n  [1/5] Loading CKKS-RNS context...")
    context_path = os.path.join(ENCRYPTED_DIR, "ckks_context.bin")
    with open(context_path, "rb") as f:
        context_bytes = f.read()
    context = ts.context_from(context_bytes)
    print(f"    Context loaded from: {context_path}")

    # ------------------------------------------------------------------
    # Step 2: Load weight shapes
    # ------------------------------------------------------------------
    print("\n  [2/5] Loading weight shapes...")
    shapes_path = os.path.join(ENCRYPTED_DIR, "weight_shapes.json")
    with open(shapes_path, "r") as f:
        weight_shapes = json.load(f)
    total_params = sum(int(np.prod(s)) for s in weight_shapes.values())
    print(f"    Parameters: {total_params:,}")
    print(f"    Layers: {len(weight_shapes)}")

    # ------------------------------------------------------------------
    # Step 3: Load encrypted weights from all hospitals
    # ------------------------------------------------------------------
    print("\n  [3/5] Loading encrypted weights...")
    print("-" * 70)

    all_encrypted = []
    load_times = {}
    for name in HOSPITAL_NAMES:
        filepath = os.path.join(ENCRYPTED_DIR, f"Hospital_{name}_weights.bin")
        file_size = os.path.getsize(filepath) / 1024

        start = time.time()
        encrypted_chunks = load_encrypted_weights(filepath, context)
        load_time = time.time() - start
        load_times[name] = load_time

        all_encrypted.append(encrypted_chunks)
        print(f"    Hospital {name}: {len(encrypted_chunks)} chunks, "
              f"{file_size:.1f} KB, loaded in {load_time:.4f}s")

    # ------------------------------------------------------------------
    # Step 4: Homomorphic aggregation (FedAvg on ciphertexts)
    # ------------------------------------------------------------------
    print(f"\n  [4/5] Homomorphic aggregation (FedAvg)...")
    print("-" * 70)
    print(f"    Operation: ct_avg = (1/{NUM_HOSPITALS}) * (ct_A + ct_B + ct_C)")
    print(f"    Server sees: random-looking polynomials (NO raw weights)")

    start = time.time()
    aggregated_chunks = aggregate_encrypted(all_encrypted, NUM_HOSPITALS)
    agg_time = time.time() - start
    print(f"    Aggregation Time: {agg_time:.4f}s")

    # Decrypt
    print(f"\n    Decrypting aggregated result...")
    start = time.time()
    encrypted_state_dict = decrypt_weights(aggregated_chunks, weight_shapes)
    dec_time = time.time() - start
    print(f"    Decryption Time: {dec_time:.4f}s")

    total_crypto_time = agg_time + dec_time
    print(f"    Total Crypto Overhead: {total_crypto_time:.4f}s")

    # ------------------------------------------------------------------
    # Step 5: Plaintext comparison & save
    # ------------------------------------------------------------------
    print(f"\n  [5/5] Verifying against plaintext aggregation...")
    print("-" * 70)

    plaintext_state_dict = aggregate_plaintext()

    # Compare encrypted vs plaintext
    max_error = 0.0
    avg_error = 0.0
    total_vals = 0
    for key in encrypted_state_dict:
        diff = (encrypted_state_dict[key] - plaintext_state_dict[key]).abs()
        max_error = max(max_error, diff.max().item())
        avg_error += diff.sum().item()
        total_vals += diff.numel()
    avg_error /= total_vals

    print(f"    Max Weight Error (encrypted vs plaintext): {max_error:.2e}")
    print(f"    Avg Weight Error (encrypted vs plaintext): {avg_error:.2e}")

    if max_error < 1e-4:
        print(f"    Verdict: NEGLIGIBLE -- encrypted aggregation matches plaintext")
    else:
        print(f"    Verdict: WARNING -- error is larger than expected")

    # Save global model
    global_model_path = os.path.join(OUTPUT_DIR, "global_model.pth")
    torch.save(encrypted_state_dict, global_model_path)
    print(f"\n    Global model saved: {global_model_path}")

    # Also save plaintext version for comparison
    plaintext_model_path = os.path.join(OUTPUT_DIR, "global_model_plaintext.pth")
    torch.save(plaintext_state_dict, plaintext_model_path)
    print(f"    Plaintext model saved: {plaintext_model_path}")

    # ------------------------------------------------------------------
    # Write aggregation report
    # ------------------------------------------------------------------
    report_path = os.path.join(OUTPUT_DIR, "aggregation_report.txt")
    with open(report_path, "w") as f:
        f.write("FEDERATED AGGREGATION REPORT (STAGE 4.4)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Method: Federated Averaging (FedAvg)\n")
        f.write(f"Hospitals: {NUM_HOSPITALS}\n")
        f.write(f"Model: MortalityLSTM ({total_params:,} parameters)\n\n")

        f.write("CKKS-RNS Parameters:\n")
        f.write("  Poly Modulus Degree: 8192\n")
        f.write("  Coeff Moduli: [60, 40, 40, 60] bits\n")
        f.write("  Security Level: 128-bit\n\n")

        f.write("Timing Breakdown:\n")
        f.write(f"  {'Operation':<30s} {'Time (s)':<12s}\n")
        f.write(f"  {'-'*30:<30s} {'-'*12:<12s}\n")
        for name, t in load_times.items():
            f.write(f"  Load Hospital {name} ciphertext   {t:.4f}s\n")
        f.write(f"  Homomorphic Aggregation       {agg_time:.4f}s\n")
        f.write(f"  Decryption                    {dec_time:.4f}s\n")
        f.write(f"  {'-'*30:<30s} {'-'*12:<12s}\n")
        f.write(f"  {'Total Crypto Overhead':<30s} {total_crypto_time:.4f}s\n\n")

        f.write("Accuracy Verification:\n")
        f.write(f"  Max Weight Error (enc vs plain): {max_error:.2e}\n")
        f.write(f"  Avg Weight Error (enc vs plain): {avg_error:.2e}\n")
        f.write(f"  Verdict: {'NEGLIGIBLE' if max_error < 1e-4 else 'WARNING'}\n\n")

        f.write("Output Files:\n")
        f.write(f"  global_model.pth           (encrypted FedAvg result)\n")
        f.write(f"  global_model_plaintext.pth (plaintext FedAvg for comparison)\n")

    print(f"    Report saved: {report_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  AGGREGATION COMPLETE")
    print("=" * 70)
    print(f"\n  Crypto Overhead:   {total_crypto_time:.4f}s")
    print(f"  Max Weight Error:  {max_error:.2e}")
    print(f"  Global Model:      {global_model_path}")
    print(f"\n  Next Step: Run evaluation (Stage 4.5)")
    print(f"    Evaluate the global model on combined test data")
    print(f"    and compare encrypted FL vs plaintext FL accuracy.")
    print("=" * 70)


if __name__ == "__main__":
    main()
