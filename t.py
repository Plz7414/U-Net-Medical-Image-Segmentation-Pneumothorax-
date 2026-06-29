import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"是否可以使用 GPU (CUDA): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"目前的 GPU 裝置: {torch.cuda.get_device_name(0)}")