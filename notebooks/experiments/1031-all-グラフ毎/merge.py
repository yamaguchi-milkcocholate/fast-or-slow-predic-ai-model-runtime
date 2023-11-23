import pandas as pd
from pathlib import Path
import itertools

rootdir = Path(__file__).resolve().parent

dfsubs = []
for arch, perm in itertools.product(["nlp", "xla"], ["default", "random"]):
    outdir = rootdir / "out" / f"{arch}-{perm}"

    dfsub = pd.read_csv(outdir / "submission_final_model.csv")
    dfsubs.append(dfsub)

dftile = dfsubs[0].loc[dfsubs[0]["ID"].str.startswith("tile:")]
dffinal = [dftile]
for i, (arch, perm) in enumerate(
    itertools.product(["nlp", "xla"], ["default", "random"])
):
    _dffinal = dfsubs[i].loc[dfsubs[i]["ID"].str.startswith(f"layout:{arch}:{perm}:")]
    dffinal.append(_dffinal)

dffinal = pd.concat(dffinal, axis=0).reset_index(drop=True)
dffinal.to_csv(rootdir / "out" / "submission_final.csv", index=False)
print(dffinal)
