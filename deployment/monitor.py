import json
import sys

import numpy as np
import pandas as pd


def engineer(raw):
    out = raw.copy()
    out["Hour"] = ((out["Time"] // 3600) % 24).astype(int)
    out["Amount_log"] = np.log1p(out["Amount"])
    return out.drop(columns=["Time", "Amount"])


def psi_report(raw_csv, ref_path="models/drift_reference.json"):
    ref = json.load(open(ref_path))
    df = engineer(pd.read_csv(raw_csv))
    rows = []
    for col, r in ref.items():
        exp = np.clip(np.array(r["expected"]), 1e-6, None)
        act = np.clip(np.histogram(df[col], r["edges"])[0] / len(df), 1e-6, None)
        rows.append((col, float(np.sum((act - exp) * np.log(act / exp)))))
    out = pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False)
    out["status"] = np.where(out.psi > 0.25, "ALERT", np.where(out.psi > 0.1, "WATCH", "OK"))
    return out


if __name__ == "__main__":
    print(psi_report(sys.argv[1]).head(10).to_string(index=False))
