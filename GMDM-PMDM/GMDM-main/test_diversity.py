import os
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

def load_mols(folder):
    mols = []
    sdf_files = [f for f in os.listdir(folder) if f.endswith(".sdf")]
    for sdf_file in sdf_files:
        path = os.path.join(folder, sdf_file)
        suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
        mol = next((m for m in suppl if m is not None), None)
        if mol:
            mols.append(mol)
    return mols

def calc_fingerprints(mols):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    return fps

def calc_diversity(fps):
    n = len(fps)
    if n < 2:
        return 0.0
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sims.append(sim)
    avg_sim = np.mean(sims)
    diversity = 1 - avg_sim
    return round(diversity, 4)

if __name__ == "__main__":
    folder = "./results/testPmolQM9"
    mols = load_mols(folder)
    fps = calc_fingerprints(mols)
    diversity_score = calc_diversity(fps)
    print(f"Diversity Score: {diversity_score}")
