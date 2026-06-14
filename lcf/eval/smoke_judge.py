"""Smoke test: confirm OPENAI_API_KEY works and the GPT-4 judge returns parseable VALID/INVALID."""
import sys
from metrics import load_env, gpt4_judge, valid_pct_gpt4

env = load_env()
key = env.get("OPENAI_API_KEY") or ""
print(f"OPENAI_API_KEY loaded: {bool(key)}  (len={len(key)}, prefix={key[:7]}...)")

examples = [
    # clearly valid
    {"premise": "All men are mortal. Socrates is a man.",
     "conclusion": "Therefore, Socrates is mortal."},
    # invalid: affirming the consequent
    {"premise": "If it rains, the ground gets wet. The ground is wet.",
     "conclusion": "Therefore, it definitely rained."},
    # invalid: faulty generalization
    {"premise": "Most successful entrepreneurs wake up early. John wakes up early.",
     "conclusion": "Therefore, John is a successful entrepreneur."},
]

verdicts = gpt4_judge(examples, model="gpt-4o", cache_path=None)
for ex, v in zip(examples, verdicts):
    print("-" * 70)
    print(f"  premise   : {ex['premise']}")
    print(f"  conclusion: {ex['conclusion']}")
    print(f"  -> valid={v['valid']}  fallacy='{v['fallacy_type']}'")
    print(f"     raw: {v['raw']!r}")

pct = sum(1 for v in verdicts if v["valid"]) / len(verdicts) * 100
print("=" * 70)
print(f"Valid%(GPT-4) over 3 examples = {pct:.1f}%  (expect ~33.3: 1 valid, 2 invalid)")
