import os
import math
import pickle
import gzip
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

_fscores = None
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

def readFragmentScores(name="fpscores.pkl.gz"):
    global _fscores
    if name == "fpscores.pkl.gz":
        name = os.path.join(os.path.dirname(__file__), name)
    data = pickle.load(gzip.open(name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculateScore(m):
    if not m.GetNumAtoms():
        return None
    if _fscores is None:
        readFragmentScores()

    sfp = mfpgen.GetSparseCountFingerprint(m)
    score1 = 0.
    nf = 0
    nze = sfp.GetNonzeroElements()
    for id, count in nze.items():
        nf += count
        score1 += _fscores.get(id, -4) * count
    score1 /= nf

    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = math.log10(2) if nMacrocycles > 0 else 0.

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    score3 = 0.
    numBits = len(nze)
    if nAtoms > numBits:
        score3 = math.log(float(nAtoms) / numBits) * .5

    sascore = score1 + score2 + score3

    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.

    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return round(sascore, 3)

def processSDFs(folder):
    readFragmentScores()
    sdf_files = [f for f in os.listdir(folder) if f.endswith(".sdf")]
    output_file = os.path.join(folder, "sa_mol.txt")
    with open(output_file, 'w') as fout:
        for sdf_file in sdf_files:
            sdf_path = os.path.join(folder, sdf_file)
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            mols = [mol for mol in suppl if mol is not None]
            if not mols:
                print(f"[空分子] {sdf_file}")
                continue
            for mol in mols:
                try:
                    Chem.SanitizeMol(mol)
                    score = calculateScore(mol)
                    fout.write(f"{sdf_file} {score}\n")
                    break
                except Exception as e:
                    print(f"[非法分子] {sdf_file}: {e}")
                    break
    print(f"所有分子已处理，结果保存在 {output_file}")

if __name__ == "__main__":
    processSDFs("./results/gen_GEOM_17iter_rdkitsc_my")