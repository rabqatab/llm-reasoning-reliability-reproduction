"""CPU smoke tests for the LCF implementation. Run with the uv venv.

  python smoke_test.py

Checks:
  1. model.py shapes flow through with d=16.
  2. each loss returns a finite scalar and total loss DECREASES over a few steps
     on synthetic separable data.
  3. extract_reps tiny hook plumbing + token alignment.
"""

import torch

from model import build_lcf
from losses import rec_loss, infonce_logic_pos, infonce_logic_neg, content_loss, total_loss


def test_shapes():
    d = 16
    lcf = build_lcf(d=d, dims=(32, 8))
    R = torch.randn(5, d)
    c, l = lcf.encode(R)
    assert c.shape == (5, 8) and l.shape == (5, 8), (c.shape, l.shape)
    lm = lcf.modify(l, sign=1)
    assert lm.shape == (5, 8)
    rec = lcf.reconstruct(c, l)
    assert rec.shape == (5, d), rec.shape
    rp = lcf(R, eta=0.5, sign=1)
    assert rp.shape == (5, d), rp.shape
    # eta controls step size
    delta = (rp - R).norm(dim=-1)
    assert torch.allclose(delta, torch.full_like(delta, 0.5), atol=1e-4), delta
    # with a non-zero V, reverse sign must differ
    lcf.set_centroids(torch.randn(8), torch.randn(8))
    rp = lcf(R, eta=0.5, sign=1)
    rm = lcf(R, eta=0.5, sign=-1)
    assert not torch.allclose(rp, rm)
    print("[shapes] OK  content/logic=(5,8) recon=(5,16) Rinput+=(5,16) ||delta||=eta")


def test_centroids():
    d = 16
    lcf = build_lcf(d=d, dims=(32, 8))
    R = torch.randn(10, d)
    _, logic = lcf.encode(R)
    labels = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    lcf.update_centroids(logic, labels)
    assert lcf.V.shape == (8,)
    assert lcf._initialized.item() == 1.0
    print("[centroids] OK  V.shape=(8,) initialized")


def test_losses_finite_and_decrease():
    torch.manual_seed(0)
    d = 16
    lcf = build_lcf(d=d, dims=(32, 8))
    # synthetic separable: valid reps near +mu, invalid near -mu
    mu = torch.randn(d) * 2.0
    N = 128
    Rp = mu + 0.3 * torch.randn(N, d)
    Rm = -mu + 0.3 * torch.randn(N, d)

    # individual losses finite
    labels = torch.cat([torch.ones(N), torch.zeros(N)]).long()
    _, logic_all = lcf.encode(torch.cat([Rp, Rm]))
    for name, val in [
        ("rec", rec_loss(lcf, torch.cat([Rp, Rm]))),
        ("logic_pos", infonce_logic_pos(logic_all, labels)),
        ("logic_neg", infonce_logic_neg(logic_all, labels)),
        ("content", content_loss(lcf, Rp, Rm)),
    ]:
        assert torch.isfinite(val), (name, val)

    opt = torch.optim.AdamW(lcf.parameters(), lr=1e-2)
    losses = []
    for step in range(60):
        lcf.update_centroids(logic_all.detach(), labels)
        loss, comps = total_loss(lcf, Rp, Rm)
        opt.zero_grad()
        loss.backward()
        opt.step()
        _, logic_all = lcf.encode(torch.cat([Rp, Rm]))
        losses.append(float(loss))
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    print(f"[losses] OK  finite; total {losses[0]:.4f} -> {losses[-1]:.4f} (decreased)")


def test_extract_tiny():
    from extract_reps import _smoke_tiny
    _smoke_tiny(list(range(10, 31)))


if __name__ == "__main__":
    test_shapes()
    test_centroids()
    test_losses_finite_and_decrease()
    test_extract_tiny()
    print("\nALL SMOKE TESTS PASSED")
