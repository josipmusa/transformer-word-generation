# Dataset
from pathlib import Path

TOKEN_SEQUENCE_LENGTH = 128

# Model
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model.pth"
LOSS_CURVE_PATH = SCRIPT_DIR / "loss_curve.png"
EMBEDDING_DIM = 384
NUM_HEADS = 8
NUM_BLOCKS = 4

# Training
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = "cuda"
