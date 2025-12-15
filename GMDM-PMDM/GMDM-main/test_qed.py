import os
from rdkit import Chem
from rdkit.Chem import QED

def processSDFs(folder):
    sdf_files = [f for f in os.listdir(folder) if f.endswith(".sdf")]
    output_file = os.path.join(folder, "qed_mol.txt")

    with open(output_file, 'w') as fout:
        for sdf_file in sdf_files:
            sdf_path = os.path.join(folder, sdf_file)
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
            mols = [mol for mol in suppl if mol is not None]
            if not mols:
                print(f"[空分子] {sdf_file}")
                continue
            for mol in mols:
                try:
                    score = QED.qed(mol)
                    fout.write(f"{sdf_file} {score:.4f}\n".replace('\\\\n', '\\n'))
                    break  # 每个文件只取第一个分子
                except Exception as e:
                    print(f"[非法分子] {sdf_file}: {e}")
                    break

    print(f"QED 分数已写入: {output_file}")

if __name__ == "__main__":
    processSDFs("./results/gen_GEOM_17iter_rdkitsc_my")