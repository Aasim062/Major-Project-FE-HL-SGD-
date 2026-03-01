"""
===============================================================================
 CKKS-RNS Homomorphic Encryption for Federated LSTM Weight Aggregation
===============================================================================

 Library:  TenSEAL (Python wrapper over Microsoft SEAL)
 Scheme:   CKKS with RNS (Residue Number System) representation
 Purpose:  Encrypt, aggregate, and decrypt LSTM model weights without
           exposing raw parameters to the aggregation server.

===============================================================================
 MATHEMATICAL FOUNDATION
===============================================================================

 1. CKKS ENCRYPTION SCHEME
 -------------------------
 CKKS (Cheon-Kim-Kim-Song, 2017) operates on polynomial rings:
 
   R = Z[X] / (X^N + 1)     where N = poly_modulus_degree (8192)
   R_q = R mod q             where q = product of coeff moduli
 
 A plaintext message m = (m_0, m_1, ..., m_{N/2-1}) ∈ C^{N/2}
 is encoded into a polynomial p(X) ∈ R via the canonical embedding:
 
   Encode(m) = Round(Δ · σ⁻¹(m))    where Δ = 2^40 (global scale)
 
 σ⁻¹ is the inverse of the canonical embedding (maps complex vectors 
 to polynomials). The scale Δ controls precision.

 2. RNS (RESIDUE NUMBER SYSTEM)
 ------------------------------
 Instead of working with huge integers mod q, RNS decomposes:
 
   q = q_0 · q_1 · q_2 · q_3    (our coeff_mod_bit_sizes = [60, 40, 40, 60])
 
 Each polynomial coefficient c is stored as:
   (c mod q_0, c mod q_1, c mod q_2, c mod q_3)
 
 This enables parallel modular arithmetic (faster than big-integer ops).
 After each multiplication, one modulus is consumed (rescaling):
   Rescale: q_0·q_1·q_2·q_3 → q_0·q_1·q_3 → q_0·q_3
 
 Our budget: 2 multiplications (enough for weighted averaging).

 3. ENCRYPTION
 -------------
 Key Generation:
   sk ← Sample from R (secret key, stays at hospital)
   pk = (a·sk + e, -a)  where a ← R_q, e ← χ (error distribution)
 
 Encrypt(m):
   ct = (pk_0·u + e_1 + Δ·m,  pk_1·u + e_2)
      = (c_0, c_1) ∈ R_q²
 
 where u ← small random polynomial, e_1, e_2 ← error distribution.

 4. HOMOMORPHIC OPERATIONS (What the server does BLINDLY)
 --------------------------------------------------------
 Addition (element-wise weight averaging):
   ct_A + ct_B = (c_0^A + c_0^B,  c_1^A + c_1^B)
 
   This works because:
   Dec(ct_A + ct_B) = Dec(ct_A) + Dec(ct_B) = w_A + w_B
 
 Scalar Multiplication (dividing by num_hospitals):
   (1/K) · ct = ((1/K)·c_0, (1/K)·c_1)
 
   Dec((1/K)·ct_sum) = (1/K)·(w_A + w_B + w_C) = w_avg
 
 5. DECRYPTION
 -------------
   Dec(ct) = c_0 + c_1·sk  (mod q, then rescale by 1/Δ)
   
   The error terms e accumulate but remain small enough for 
   approximate correctness (CKKS guarantee):
   
   |Dec(ct) - m| < ε    where ε ≈ 2^{-20} for our parameters

 6. SECURITY GUARANTEE
 ---------------------
 Based on Ring-LWE (Ring Learning With Errors):
   Given (a, a·sk + e), finding sk is computationally infeasible.
   
   Security level: 128-bit (with N=8192, our parameter choice)
   
   The server sees only (c_0, c_1) — random-looking polynomials.
   Without sk, recovering the weights w is as hard as solving Ring-LWE.

===============================================================================
 FEDERATED AGGREGATION WORKFLOW
===============================================================================

 Round r of Federated Learning:

 Hospital A:  w_A^(r) = LocalTrain(w_global^(r-1), D_A)
 Hospital B:  w_B^(r) = LocalTrain(w_global^(r-1), D_B)  
 Hospital C:  w_C^(r) = LocalTrain(w_global^(r-1), D_C)

 Each hospital encrypts:
   ct_A = Enc(w_A^(r)),  ct_B = Enc(w_B^(r)),  ct_C = Enc(w_C^(r))

 Server computes (without decrypting):
   ct_sum = ct_A ⊕ ct_B ⊕ ct_C           (homomorphic addition)
   ct_avg = (1/3) ⊗ ct_sum                (homomorphic scalar mult)

 Each hospital decrypts:
   w_global^(r) = Dec(ct_avg)

 Mathematical guarantee:
   w_global^(r) = (1/3)(w_A + w_B + w_C) + ε
   where |ε| ≈ 10^{-7} (negligible noise)

===============================================================================
"""

import tenseal as ts
import torch
import numpy as np
import time
import os
import sys

# Add LSTM model path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Model_Training'))
from LSTM import MortalityLSTM

# ============================================================
# CKKS-RNS CONTEXT CREATION
# ============================================================
def create_ckks_context():
    """
    Creates a TenSEAL CKKS context with RNS representation.
    
    Parameters explained:
    - poly_modulus_degree = 8192 (N)
        Ring dimension. Must be power of 2.
        N=8192 gives 128-bit security with our moduli.
        Supports N/2 = 4096 slots per ciphertext.
        
    - coeff_mod_bit_sizes = [60, 40, 40, 60]
        RNS decomposition of ciphertext modulus q.
        q = q_0 (60 bits) × q_1 (40 bits) × q_2 (40 bits) × q_3 (60 bits)
        
        First/Last (60-bit): Special primes for encoding/decoding precision.
        Middle (40-bit): Consumed during rescaling after multiplications.
        Number of middle primes = max multiplicative depth = 2.
        
    - global_scale = 2^40
        Δ (Delta) = scaling factor for encoding floats into integers.
        Larger Δ = more precision but uses more noise budget.
        2^40 ≈ 10^12, giving ~12 decimal digits of precision.
    
    Returns:
        ts.Context with CKKS scheme configured
    """
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    
    # Set the global scale (Δ = 2^40)
    context.global_scale = 2**40
    
    # Generate Galois keys (needed for rotations, not strictly required 
    # for simple addition but included for completeness)
    context.generate_galois_keys()
    
    print("CKKS-RNS Context Created:")
    print(f"  Polynomial Modulus Degree (N): 8192")
    print(f"  Coefficient Moduli: [60, 40, 40, 60] bits")
    print(f"  Global Scale (Δ): 2^40 = {2**40:,}")
    print(f"  Max Slots per Ciphertext: {8192 // 2}")
    print(f"  Security Level: 128-bit")
    print(f"  Max Multiplicative Depth: 2")
    
    return context

# ============================================================
# WEIGHT EXTRACTION & RECONSTRUCTION
# ============================================================
def flatten_model_weights(model):
    """
    Extracts all model parameters and flattens into a single 1D list.
    
    MortalityLSTM has 5,921 parameters across 8 tensors:
      lstm.weight_ih_l0:    (128, 8)  = 1024
      lstm.weight_hh_l0:    (128, 32) = 4096
      lstm.bias_ih_l0:      (128,)    = 128
      lstm.bias_hh_l0:      (128,)    = 128
      classifier.0.weight:  (16, 32)  = 512
      classifier.0.bias:    (16,)     = 16
      classifier.3.weight:  (1, 16)   = 16
      classifier.3.bias:    (1,)      = 1
                                 Total: 5,921
    
    Returns:
        flat_weights: list of 5,921 float values
        weight_shapes: dict mapping parameter name → shape (for reconstruction)
    """
    flat_weights = []
    weight_shapes = {}
    
    for name, param in model.state_dict().items():
        weight_shapes[name] = param.shape
        flat_weights.extend(param.cpu().numpy().flatten().tolist())
    
    return flat_weights, weight_shapes

def reconstruct_model_weights(flat_weights, weight_shapes):
    """
    Reconstructs a state_dict from a flat list of weights.
    Inverse of flatten_model_weights().
    
    Args:
        flat_weights: list of 5,921 floats (decrypted)
        weight_shapes: dict of {param_name: torch.Size}
    
    Returns:
        state_dict: OrderedDict compatible with model.load_state_dict()
    """
    state_dict = {}
    offset = 0
    
    for name, shape in weight_shapes.items():
        num_params = 1
        for s in shape:
            num_params *= s
        
        # Slice the flat array and reshape
        param_flat = flat_weights[offset:offset + num_params]
        param_tensor = torch.tensor(param_flat, dtype=torch.float32).reshape(shape)
        state_dict[name] = param_tensor
        offset += num_params
    
    return state_dict

# ============================================================
# ENCRYPT MODEL WEIGHTS
# ============================================================
def encrypt_weights(model, context):
    """
    Encrypts all model weights into a CKKS ciphertext vector.
    
    Mathematical operation:
        w = [w_0, w_1, ..., w_5920]   (plaintext weights)
        
        Encode: p(X) = Round(Δ · σ⁻¹(w))
        Encrypt: ct = (pk_0·u + e_1 + p(X), pk_1·u + e_2)
    
    The 5,921 weights fit in one ciphertext (max slots = 4096).
    Since 5,921 > 4096, we split into chunks.
    
    Returns:
        encrypted_chunks: list of ts.CKKSVector (encrypted weight chunks)
        weight_shapes: dict for reconstruction
        encryption_time: float (seconds)
    """
    start_time = time.time()
    
    flat_weights, weight_shapes = flatten_model_weights(model)
    total_params = len(flat_weights)
    
    # CKKS with N=8192 supports N/2 = 4096 slots per ciphertext
    max_slots = 8192 // 2  # 4096
    
    # Split weights into chunks that fit in one ciphertext
    encrypted_chunks = []
    for i in range(0, total_params, max_slots):
        chunk = flat_weights[i:i + max_slots]
        encrypted_chunk = ts.ckks_vector(context, chunk)
        encrypted_chunks.append(encrypted_chunk)
    
    encryption_time = time.time() - start_time
    
    print(f"  Encrypted {total_params} weights into {len(encrypted_chunks)} ciphertext(s)")
    print(f"  Encryption Time: {encryption_time:.4f}s")
    
    return encrypted_chunks, weight_shapes, encryption_time

# ============================================================
# DECRYPT MODEL WEIGHTS
# ============================================================
def decrypt_weights(encrypted_chunks, weight_shapes, context=None):
    """
    Decrypts CKKS ciphertext vectors back into model weights.
    
    Mathematical operation:
        Dec(ct) = c_0 + c_1 · sk   (mod q)
        Decode: w' = (1/Δ) · σ(Dec(ct))
        
        |w' - w| < ε ≈ 10^{-7}   (CKKS approximate guarantee)
    
    Returns:
        state_dict: reconstructed model weights
        decryption_time: float (seconds)
    """
    start_time = time.time()
    
    # Decrypt all chunks and concatenate
    flat_weights = []
    for chunk in encrypted_chunks:
        decrypted = chunk.decrypt()
        flat_weights.extend(decrypted)
    
    # Trim to exact number of parameters (last chunk may have padding)
    total_params = sum(1 for shape in weight_shapes.values() for _ in [np.prod(list(shape))])
    total_params = sum(int(np.prod(list(shape))) for shape in weight_shapes.values())
    flat_weights = flat_weights[:total_params]
    
    # Reconstruct state dict
    state_dict = reconstruct_model_weights(flat_weights, weight_shapes)
    
    decryption_time = time.time() - start_time
    print(f"  Decrypted {total_params} weights")
    print(f"  Decryption Time: {decryption_time:.4f}s")
    
    return state_dict, decryption_time

# ============================================================
# HOMOMORPHIC AGGREGATION (Server-side, NO decryption)
# ============================================================
def aggregate_encrypted(list_of_encrypted_chunks, num_hospitals):
    """
    Performs Federated Averaging on encrypted weights.
    The server NEVER sees the actual weight values.
    
    Mathematical operation:
        Given K hospitals with encrypted weights:
          ct_1, ct_2, ..., ct_K
        
        Step 1 — Homomorphic Addition:
          ct_sum = ct_1 ⊕ ct_2 ⊕ ... ⊕ ct_K
          
          In ring arithmetic:
          ct_sum = (Σ c_0^i, Σ c_1^i)
          
          Correctness: Dec(ct_sum) = Σ w_i + Σ e_i
        
        Step 2 — Homomorphic Scalar Multiplication:
          ct_avg = (1/K) ⊗ ct_sum
          
          In ring arithmetic:
          ct_avg = ((1/K)·c_0_sum, (1/K)·c_1_sum)
          
          Correctness: Dec(ct_avg) = (1/K)·Σ w_i + (1/K)·Σ e_i
                                    = w_avg + ε
          
          where |ε| ≈ K · 2^{-40} ≈ 10^{-12} (negligible)
    
    Args:
        list_of_encrypted_chunks: list of [encrypted_chunks] per hospital
            Each hospital contributes a list of ciphertext chunks
        num_hospitals: K (number of hospitals to average)
    
    Returns:
        aggregated_chunks: list of averaged encrypted chunks
        aggregation_time: float (seconds)
    """
    start_time = time.time()
    
    num_chunks = len(list_of_encrypted_chunks[0])
    aggregated_chunks = []
    
    for chunk_idx in range(num_chunks):
        # Step 1: Homomorphic Addition — ct_sum = Σ ct_i
        summed = list_of_encrypted_chunks[0][chunk_idx]
        
        for hospital_idx in range(1, num_hospitals):
            summed = summed + list_of_encrypted_chunks[hospital_idx][chunk_idx]
        
        # Step 2: Homomorphic Scalar Multiplication — ct_avg = (1/K) · ct_sum
        averaged = summed * (1.0 / num_hospitals)
        
        aggregated_chunks.append(averaged)
    
    aggregation_time = time.time() - start_time
    
    print(f"  Aggregated {num_hospitals} hospitals' weights ({num_chunks} chunks)")
    print(f"  Aggregation Time: {aggregation_time:.4f}s")
    print(f"  Operations: {num_chunks} additions + {num_chunks} scalar multiplications")
    print(f"  Server saw: ZERO plaintext values (fully blind)")
    
    return aggregated_chunks, aggregation_time

# ============================================================
# PRECISION VERIFICATION
# ============================================================
def verify_encryption_precision(model, context):
    """
    Verifies that encrypt → decrypt introduces negligible error.
    
    Tests: |w_original - w_decrypted| for all 5,921 parameters
    """
    print("\n=== PRECISION VERIFICATION ===")
    
    # Original weights
    original_flat, shapes = flatten_model_weights(model)
    
    # Encrypt → Decrypt
    enc_chunks, shapes, enc_time = encrypt_weights(model, context)
    dec_state_dict, dec_time = decrypt_weights(enc_chunks, shapes)
    
    # Compare
    decrypted_flat = []
    for name in shapes:
        decrypted_flat.extend(dec_state_dict[name].numpy().flatten().tolist())
    
    original_np = np.array(original_flat)
    decrypted_np = np.array(decrypted_flat)
    
    abs_error = np.abs(original_np - decrypted_np)
    
    print(f"\n  Total Parameters: {len(original_flat)}")
    print(f"  Max Absolute Error:  {abs_error.max():.2e}")
    print(f"  Mean Absolute Error: {abs_error.mean():.2e}")
    print(f"  Relative Error:     {(abs_error / (np.abs(original_np) + 1e-10)).mean():.2e}")
    print(f"  Encryption Time:    {enc_time:.4f}s")
    print(f"  Decryption Time:    {dec_time:.4f}s")
    
    return abs_error.max(), abs_error.mean()

# ============================================================
# DEMO: Full Encrypt → Aggregate → Decrypt Pipeline
# ============================================================
def demo_full_pipeline():
    """
    Demonstrates the complete CKKS-RNS workflow:
    1. Create 3 models (simulating 3 hospitals)
    2. Encrypt each model's weights
    3. Aggregate encrypted weights (server-side, blind)
    4. Decrypt aggregated weights
    5. Verify result matches plaintext averaging
    """
    print("=" * 60)
    print("  CKKS-RNS HOMOMORPHIC ENCRYPTION DEMO")
    print("  Federated Averaging of 3 Hospital LSTM Models")
    print("=" * 60)
    
    # Step 0: Create CKKS Context
    print("\n--- Step 0: CKKS-RNS Context Setup ---")
    context = create_ckks_context()
    
    # Step 1: Create 3 hospital models with different random weights
    print("\n--- Step 1: Simulating 3 Hospital Models ---")
    num_hospitals = 3
    hospitals = []
    
    for i in range(num_hospitals):
        torch.manual_seed(i * 42)  # Different random init per hospital
        model = MortalityLSTM()
        hospitals.append(model)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Hospital {chr(65+i)}: {params} parameters initialized")
    
    # Step 2: Each hospital encrypts its weights
    print("\n--- Step 2: Hospitals Encrypt Weights ---")
    all_encrypted = []
    weight_shapes = None
    total_enc_time = 0
    
    for i, model in enumerate(hospitals):
        print(f"\n  Hospital {chr(65+i)}:")
        enc_chunks, shapes, enc_time = encrypt_weights(model, context)
        all_encrypted.append(enc_chunks)
        weight_shapes = shapes  # Same structure for all
        total_enc_time += enc_time
    
    # Step 3: Server aggregates (BLIND — no decryption)
    print("\n--- Step 3: Server Aggregates (Homomorphic, Blind) ---")
    agg_chunks, agg_time = aggregate_encrypted(all_encrypted, num_hospitals)
    
    # Step 4: Decrypt aggregated weights
    print("\n--- Step 4: Decrypt Aggregated Weights ---")
    agg_state_dict, dec_time = decrypt_weights(agg_chunks, weight_shapes)
    
    # Step 5: Verify against plaintext averaging
    print("\n--- Step 5: Verification (Encrypted vs Plaintext Average) ---")
    
    # Compute plaintext average
    plaintext_avg = {}
    for name in weight_shapes:
        avg_param = torch.zeros(weight_shapes[name])
        for model in hospitals:
            avg_param += model.state_dict()[name]
        avg_param /= num_hospitals
        plaintext_avg[name] = avg_param
    
    # Compare
    max_error = 0
    total_error = 0
    total_params = 0
    
    for name in weight_shapes:
        encrypted_val = agg_state_dict[name].numpy()
        plaintext_val = plaintext_avg[name].numpy()
        error = np.abs(encrypted_val - plaintext_val)
        max_error = max(max_error, error.max())
        total_error += error.sum()
        total_params += error.size
    
    mean_error = total_error / total_params
    
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total Parameters per Model:  {total_params}")
    print(f"  Number of Hospitals:         {num_hospitals}")
    print(f"  Max |Encrypted - Plaintext|: {max_error:.2e}")
    print(f"  Mean |Encrypted - Plaintext|:{mean_error:.2e}")
    print(f"  Encryption Time (total):     {total_enc_time:.4f}s")
    print(f"  Aggregation Time:            {agg_time:.4f}s")
    print(f"  Decryption Time:             {dec_time:.4f}s")
    print(f"  Total Overhead:              {total_enc_time + agg_time + dec_time:.4f}s")
    print(f"{'=' * 60}")
    
    if max_error < 1e-3:
        print("  ✓ VERIFIED: Encrypted aggregation matches plaintext averaging!")
        print(f"  ✓ Error is negligible ({max_error:.2e} << 0.001)")
    else:
        print("  ✗ WARNING: Error exceeds threshold. Check parameters.")
    
    print(f"{'=' * 60}")
    
    return context, agg_state_dict, weight_shapes

# ============================================================
# LOAD .PTH FILE → ENCRYPT (REAL TRAINED WEIGHTS)
# ============================================================
def encrypt_from_pth(pth_path, context=None):
    """
    Demonstrates how a .pth file gets encrypted step by step.
    
    .pth Flow Explained:
    ─────────────────────
    
    STEP 1 — .pth file is a serialized Python dict saved by PyTorch:
        torch.save(model.state_dict(), "lstm_baseline.pth")
        
        It contains: {
            "lstm.weight_ih_l0": tensor of shape (128, 8),   ← 1024 floats
            "lstm.weight_hh_l0": tensor of shape (128, 32),  ← 4096 floats
            "lstm.bias_ih_l0":   tensor of shape (128,),     ← 128 floats
            ...                                               Total: 5921 floats
        }
    
    STEP 2 — torch.load() deserializes it back into PyTorch tensors (in RAM):
        state_dict = torch.load("lstm_baseline.pth")
        model.load_state_dict(state_dict)
        
        Now model.parameters() are the trained float values:
        e.g., lstm.weight_ih_l0[0][0] = -0.03421...
    
    STEP 3 — Flatten all tensors into a single list of 5,921 Python floats:
        flat_weights = [-0.0342, 0.1205, -0.0891, ...]
    
    STEP 4 — CKKS Encryption encodes these floats into polynomials:
        Each float w_i is scaled: round(w_i × 2^40)
        Mapped into polynomial ring R_q = Z[X]/(X^8192 + 1)
        Encrypted with public key → ciphertext (c_0, c_1)
        
        The ciphertext is random-looking numbers that hide the weights.
    
    Args:
        pth_path: path to .pth file
        context: TenSEAL context (created if None)
    
    Returns:
        encrypted_chunks, weight_shapes, context
    """
    print("=" * 60)
    print("  ENCRYPTING WEIGHTS FROM .PTH FILE")
    print("=" * 60)
    
    # === STEP 1: Read .pth file from disk ===
    pth_size = os.path.getsize(pth_path)
    print(f"\n  STEP 1 — Read .pth file:")
    print(f"    Path: {pth_path}")
    print(f"    File size: {pth_size:,} bytes ({pth_size/1024:.1f} KB)")
    print(f"    Format: PyTorch serialized state_dict (pickle + tensor data)")
    
    # === STEP 2: Load into PyTorch model ===
    print(f"\n  STEP 2 — torch.load() → PyTorch tensors in RAM:")
    model = MortalityLSTM()
    state_dict = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"    Loaded {len(state_dict)} weight tensors:")
    for name, param in state_dict.items():
        print(f"      {name:30s} shape={str(list(param.shape)):15s} "
              f"→ {param.numel():5d} floats  "
              f"(range: [{param.min():.4f}, {param.max():.4f}])")
    
    # === STEP 3: Flatten to Python list ===
    print(f"\n  STEP 3 — Flatten all tensors into 1D list:")
    flat_weights, weight_shapes = flatten_model_weights(model)
    print(f"    Total floats: {len(flat_weights)}")
    print(f"    First 5 values: {[f'{w:.6f}' for w in flat_weights[:5]]}")
    print(f"    Last 5 values:  {[f'{w:.6f}' for w in flat_weights[-5:]]}")
    
    # === STEP 4: CKKS Encryption ===
    print(f"\n  STEP 4 — CKKS-RNS Encryption:")
    if context is None:
        context = create_ckks_context()
    
    enc_chunks, shapes, enc_time = encrypt_weights(model, context)
    
    print(f"\n    Before encryption: {len(flat_weights)} readable float values")
    print(f"    After encryption:  {len(enc_chunks)} opaque ciphertext object(s)")
    print(f"    The server can ONLY see encrypted ciphertexts, NOT the floats.")
    
    # === Verify round-trip ===
    print(f"\n  STEP 5 — Verify decrypt recovers original:")
    dec_state_dict, dec_time = decrypt_weights(enc_chunks, shapes)
    
    max_err = 0
    for name in shapes:
        err = (state_dict[name].cpu().float() - dec_state_dict[name]).abs().max().item()
        max_err = max(max_err, err)
    
    print(f"    Max error after encrypt→decrypt: {max_err:.2e}")
    print(f"    ✓ Original weights perfectly recoverable!" if max_err < 1e-5 else 
          f"    ✗ Error too large!")
    
    print(f"\n{'=' * 60}")
    
    return enc_chunks, weight_shapes, context

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    
    # ---- Demo 1: Encrypt actual trained .pth file ----
    PTH_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'models', 'lstm_baseline.pth'
    )
    
    if os.path.exists(PTH_PATH):
        enc_chunks, shapes, context = encrypt_from_pth(PTH_PATH)
    else:
        print(f"[!] .pth file not found at {PTH_PATH}, using random weights")
        context = create_ckks_context()
    
    # ---- Demo 2: Full 3-hospital federated pipeline ----
    print("\n\n")
    context, agg_weights, shapes = demo_full_pipeline()
    
    # ---- Demo 3: Single-model precision test ----
    print("\n\n")
    model = MortalityLSTM()
    verify_encryption_precision(model, context)
