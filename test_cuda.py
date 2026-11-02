import torch
print("CUDA Available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0), flush=True)
    print("Memory:", torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB", flush=True)
