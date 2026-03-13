import torch

MODEL_PATH = "oxononet.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
