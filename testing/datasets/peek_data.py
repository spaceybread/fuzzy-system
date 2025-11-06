import polars as pl
import numpy as np
from pathlib import Path

root_dirs = ["voice_128", "voice_64", "ytf_128", "ytf_64"]

zero_map = {
    "voice_128": -10.8454900509,
    "voice_64": -10.8454900509,
    "ytf_128": -11.6393407087,
    "ytf_64": -11.6393407087,
}

models = ["radial", "e8", "gauss_0", "gauss_1", "gauss_2"]
temps = [70, 90]

tables = {70: {}, 90: {}}

for name in root_dirs:
    res_dir = Path(name) / "tests" / "results"
    for temp in temps:
        row = {}
        for model in models:
            file = res_dir / f"{model}_{temp}.csv"
            if not file.exists():
                row[model] = None
                continue
            df = pl.read_csv(file)
            fmr = df["FMR"][0]
            if fmr > 0:
                val = np.log2(fmr)
            else:
                val = zero_map[name]
            row[model] = val
        tables[temp][name] = row

table_70 = pl.DataFrame(
    {**{"dir": list(tables[70].keys())},
     **{m: [tables[70][d][m] for d in tables[70]] for m in models}}
).set_sorted("dir")

table_90 = pl.DataFrame(
    {**{"dir": list(tables[90].keys())},
     **{m: [tables[90][d][m] for d in tables[90]] for m in models}}
).set_sorted("dir")

print("=== Table 70 ===")
print(table_70)

table_70.write_csv("full_results/table_70.csv")

print("\n=== Table 90 ===")
print(table_90)

table_90.write_csv("full_results/table_90.csv")
