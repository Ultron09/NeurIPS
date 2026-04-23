# Appendix: Formal Algorithmic Logic (ANTARA V9.4)

## Algorithm 1: Task Regime Classification (TRC)
**Goal**: Dynamically adjust framework plasticity based on uncertainty-weighted priors.

```latex
Algorithm 1: Task Regime Classification (TRC)
------------------------------------------------------------
Input: Current prediction error E, latent entropy H(z), task fingerprint Phi, vault index V
Output: Regime R, Meta-LR eta_meta

1. Initialize uncertainty buffer Sigma = {E_t-k, ..., E_t}
2. Calculate Task Similarity: S_v = max_k [ cosine_sim(Phi, V_k) ]
3. Calculate Cognitive Surprise: S_c = | E - Mean(Sigma) | / Std(Sigma)
4. IF S_v > tau_similarity THEN
     R = TRANSFER
     eta_meta = eta_base * (1 - S_v)
   ELSE IF S_c > tau_surprise AND H(z) > tau_entropy THEN
     R = SCRATCH
     eta_meta = eta_max
   ELSE IF len(V) > 0 THEN
     R = CONTINUOUS
     eta_meta = eta_stable
   ELSE
     R = DISTILLATION (GHOST)
     eta_meta = eta_min
5. RETURN R, eta_meta
```

## Algorithm 2: Cognitive Anchor & Shift (CAS) Update
**Goal**: Enforce parameter immutability on core manifolds to guarantee BWT=0.

```latex
Algorithm 2: CAS-Protected Parameter Update
------------------------------------------------------------
Input: Model parameters theta, Gradients g = nabla_theta L, 
       Importance Mask M_cas in {0,1}^d, Learning rate eta
Output: Updated parameters theta_next

1. Identify Sacred Core: M_cas = {j : omega_j > tau_importance}
2. Capture Gradient: g = compute_gradients(L, theta)
3. Apply Gradient Shunting:
   FOR each parameter j in d:
     IF M_cas[j] == 1 THEN
       g_j = 0  // Shunt gradient to zero for core manifold
     ELSE
       g_j = g_j // Allow shift in non-critical space
4. Update Step: theta_next = theta - eta * g
5. Project to Orthogonal Space (Optional):
   IF R == CONTINUOUS:
     theta_next = OGD_Project(theta_next, subspace_proj)
6. RETURN theta_next
```

## Algorithm 3: JIT Expert Retrieval & Vault Management
**Goal**: Manage O(1) VRAM scaling via off-device parameter offloading.

```latex
Algorithm 3: JIT Expert Retrieval & Vault Management
------------------------------------------------------------
Input: Latent features z, Vault V on Host_CPU, GPU_Memory_Limit L_vram
Output: Active Expert Block E_k

1. Generate Task Fingerprint: Phi = GlobalAveragePool(z)
2. Search Vault: k = argmax_i [ sim(Phi, V_i.fingerprint) ]
3. IF sim(Phi, V_k.fingerprint) > tau_retrieval THEN
     // JIT Retrieval Logic
     IF gpu_allocated(E_active) != E_k THEN
       OffloadToCPU(E_active)
       LoadToGPU(V_k.parameters)
       E_active = E_k
   ELSE
     // Expansion Logic
     IF VRAM_Usage > L_vram THEN
       ConsolidateAndOffload(argmin_t Usage(t))
     E_active = SpawnNewExpert(FiLM_Modulation)
4. Forward Pass: y = E_active(z, gamma_k, beta_k)
5. RETURN E_active, y
```
