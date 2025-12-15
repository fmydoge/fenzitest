import torch
  
    # 替换为您的检查点文件的完整路径
checkpoint_path = './logs/crossdock_exp_2025_08_28__10_37_21/checkpoints/500.pt'
     
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("检查点文件中的键 (keys):")
    for key in checkpoint.keys():
        print(f"- {key}")
except Exception as e:
    print(f"加载检查点文件时发生错误: {e}")
