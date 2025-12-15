import os
from rdkit import Chem
from rdkit.Chem import rdchem

# 目标目录
sdf_folder = r"./results/testPmolQM9"

# 收集所有 SDF 文件
sdf_files = [f for f in os.listdir(sdf_folder) if f.endswith(".sdf")]

# 记录非法文件
invalid_files = []

for sdf_file in sdf_files:
    sdf_path = os.path.join(sdf_folder, sdf_file)
    try:
        # 读取分子对象
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        mols = [mol for mol in suppl if mol is not None]

        if not mols:
            print(f"[空分子] {sdf_file}")
            invalid_files.append(sdf_file)
            continue

        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"[非法分子] {sdf_file}: {e}")
                invalid_files.append(sdf_file)
                break  # 若文件中有一个非法分子，就跳过整个文件
    except Exception as e:
        print(f"[解析失败] {sdf_file}: {e}")
        invalid_files.append(sdf_file)

# 打印结果
print("\n====== 检测完成 ======")
print(f"总文件数：{len(sdf_files)}")
print(f"非法文件数：{len(invalid_files)}")

if invalid_files:
    print("\n以下文件包含非法分子：")
    for fname in invalid_files:
        print(f"- {fname}")
else:
    print("所有分子都合法 ✅")
