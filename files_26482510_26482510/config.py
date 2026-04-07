import torch

MODEL_PATH = "files_26482510_NOMA2/oxononet.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
