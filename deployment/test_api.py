import json
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
fraud = json.load(open("sample_fraud.json"))
legit = json.load(open("sample_legit.json"))

r = requests.get(f"{BASE}/health", timeout=15)
assert r.status_code == 200 and r.json()["model_loaded"], r.text
print("health OK")

out = {}
for name, tx in [("fraud", fraud), ("legit", legit)]:
    r = requests.post(f"{BASE}/predict", json=tx, timeout=15)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"fraud_probability", "decision", "risk_level"} <= body.keys()
    out[name] = body
    print(name, "->", body)
assert out["fraud"]["fraud_probability"] > out["legit"]["fraud_probability"]

r = requests.post(f"{BASE}/predict/batch", json=[fraud, legit], timeout=15)
assert r.status_code == 200 and r.json()["count"] == 2, r.text
print("batch OK")

bad = dict(legit)
bad.pop("V1")
r = requests.post(f"{BASE}/predict", json=bad, timeout=15)
assert r.status_code == 422, "missing field should be rejected"
print("validation OK")

print("ALL API TESTS PASSED")
