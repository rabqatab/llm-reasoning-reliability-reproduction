"""Local CPU unit test for the LCF module (no GPU / no LLM needed).

Synthetic setup: each content vector c lives in R^d. A fixed unit "logic axis"
u encodes validity:  valid = c + a*u,  invalid = c - a*u.
A correct LCF must (1) drive all losses down, (2) reconstruct well, and
(3) when steered, push an INVALID rep toward the VALID region (its component
along u must increase). This validates wiring, losses, steering, and modify().

Run:  python tests/test_lcf_local.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from lcf import LCF

torch.manual_seed(0)
D = 32          # tiny "d_model"
K = 6           # content lives in a low-rank subspace (reconstructable from bottleneck)
N = 512         # pairs
A = 3.0         # logic separation magnitude

# Orthonormal basis: first K dims span content, dim K is the logic axis u.
Q, _ = torch.linalg.qr(torch.randn(D, D))
basis = Q[:K]                                      # (K, D) content subspace
u = Q[K]                                           # logic axis, orthogonal to content
content = torch.randn(N, K) @ basis                # (N, D), rank K
r_valid = content + A * u
r_invalid = content - A * u

model = LCF(d_model=D, proj_hidden=16, proj_out=8, dec_hidden=16, tau=0.2)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)

first = None
for step in range(800):
    idx = torch.randperm(N)[:128]
    opt.zero_grad()
    total, parts = model.losses(r_valid[idx], r_invalid[idx])
    total.backward()
    opt.step()
    if step == 0:
        first = parts["total"]
    if step % 200 == 0:
        print(f"step {step:4d}  total={parts['total']:.4f}  "
              f"rec={parts['rec']:.4f}  logic={parts['logic']:.4f}  content={parts['content']:.4f}")
last = parts["total"]

# --- set steering vector from trained logic reps -----------------------------
with torch.no_grad():
    _, zv = model.project(r_valid)
    _, zi = model.project(r_invalid)
model.set_steering_vector(zv, zi)

# --- measure logic-space separation (the contrastive loss's actual job) -------
import torch.nn.functional as F
with torch.no_grad():
    zvn, zin = F.normalize(zv, dim=-1), F.normalize(zi, dim=-1)
    within = ((zvn @ zvn.t()).mean() + (zin @ zin.t()).mean()).item() / 2
    across = (zvn @ zin.t()).mean().item()
    center_cos = F.cosine_similarity(zvn.mean(0), zin.mean(0), dim=0).item()

# --- steer invalid reps toward valid; component along u must increase ---------
with torch.no_grad():
    before = (r_invalid @ u).mean().item()
    steered = model.modify(r_invalid.unsqueeze(1), eta=0.5).squeeze(1)
    after = (steered @ u).mean().item()
    # reverse steering (invalid modification) must go the other way
    steered_inv = model.modify(r_valid.unsqueeze(1), eta=0.5, sign=-1.0).squeeze(1)
    after_inv = (steered_inv @ u).mean().item()
    before_v = (r_valid @ u).mean().item()
    rec_mse = torch.nn.functional.mse_loss(model.reconstruct(r_valid), r_valid).item()

# --- inference shape check over a sequence (B,T,d) ---------------------------
seq = torch.randn(2, 7, D)
out = model.modify(seq, eta=0.5)
assert out.shape == seq.shape, f"seq shape mismatch: {out.shape}"

print("\n--- results ---")
print(f"loss {first:.3f} -> {last:.3f}  (rec/content collapsed; logic floor = log(n_pos))")
print(f"reconstruction MSE (valid): {rec_mse:.4f}   content loss: {parts['content']:.4f}")
print(f"logic-space cos: within-class {within:+.3f}  across-class {across:+.3f}  centers {center_cos:+.3f}")
print(f"invalid rep, <.,u>: {before:+.3f} -> steered {after:+.3f}  (valid baseline {before_v:+.3f})")
print(f"valid rep reverse-steered, <.,u>: {before_v:+.3f} -> {after_inv:+.3f}")

ok = True
def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg); return cond
ok &= check(rec_mse < 0.05, "reconstruction (L_rec)")
ok &= check(parts['content'] < 0.05, "content disentangle (L_content)")
ok &= check(within - across > 0.5 and center_cos < 0.5, "logic-space separation (contrastive)")
ok &= check(after > before + 0.3, "steering -> valid region (eq 1-5)")
ok &= check(after_inv < before_v - 0.3, "reverse steering / invalid modification")
print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
