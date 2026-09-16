import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"..\dataset_split"

TRAIN_IR_DIR = os.path.join(
    BASE_DIR,
    "train",
    "infrared"
)

TRAIN_RGB_DIR = os.path.join(
    BASE_DIR,
    "train",
    "rgb"
)

VAL_IR_DIR = os.path.join(
    BASE_DIR,
    "val",
    "infrared"
)

VAL_RGB_DIR = os.path.join(
    BASE_DIR,
    "val",
    "rgb"
)

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "colorization_model.pth"
)

LOSS_PLOT_PATH = os.path.join(
    MODEL_DIR,
    "training_loss.png"
)

IMAGE_SIZE = 256

# CPU-friendly settings
BATCH_SIZE = 1

# Total number of epochs
NUM_EPOCHS = 20

LEARNING_RATE = 0.0002

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# DATASET
# ============================================================

class ColorizationDataset(Dataset):

    def __init__(
        self,
        ir_dir,
        rgb_dir
    ):

        self.ir_dir = ir_dir
        self.rgb_dir = rgb_dir

        self.ir_files = sorted([
            f
            for f in os.listdir(ir_dir)
            if f.lower().endswith(".png")
        ])

        # Keep only files having matching RGB images
        self.ir_files = [
            f
            for f in self.ir_files
            if os.path.exists(
                os.path.join(
                    rgb_dir,
                    f
                )
            )
        ]

        print(
            f"Loaded {len(self.ir_files)} paired images "
            f"from {ir_dir}"
        )


    def __len__(self):

        return len(self.ir_files)


    def __getitem__(self, index):

        filename = self.ir_files[index]

        ir_path = os.path.join(
            self.ir_dir,
            filename
        )

        rgb_path = os.path.join(
            self.rgb_dir,
            filename
        )


        # ----------------------------------------------------
        # Load IR
        # ----------------------------------------------------

        ir = Image.open(
            ir_path
        ).convert("L")


        # ----------------------------------------------------
        # Load RGB
        # ----------------------------------------------------

        rgb = Image.open(
            rgb_path
        ).convert("RGB")


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        ir = ir.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.BILINEAR
        )

        rgb = rgb.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.BILINEAR
        )


        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        ir = np.array(
            ir
        ).astype(
            np.float32
        ) / 255.0

        rgb = np.array(
            rgb
        ).astype(
            np.float32
        ) / 255.0


        # ----------------------------------------------------
        # Convert to Tensor
        # ----------------------------------------------------

        # IR:
        # H x W → 1 x H x W

        ir = torch.from_numpy(
            ir
        ).unsqueeze(0)


        # RGB:
        # H x W x 3 → 3 x H x W

        rgb = torch.from_numpy(
            rgb.transpose(
                2,
                0,
                1
            )
        )


        return ir, rgb


# ============================================================
# DOUBLE CONVOLUTION BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


# ============================================================
# LIGHTWEIGHT U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.enc1 = DoubleConv(
            1,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )


        # ----------------------------------------------------
        # Pooling
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            2
        )


        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(
            128,
            256
        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )


        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )


        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )


        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        self.final = nn.Conv2d(
            32,
            3,
            kernel_size=1
        )


    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )


        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = self.bottleneck(
            self.pool(e3)
        )


        # ====================================================
        # DECODER
        # ====================================================

        d3 = self.up3(b)

        d3 = torch.cat(
            [
                d3,
                e3
            ],
            dim=1
        )

        d3 = self.dec3(d3)


        # ----------------------------------------------------

        d2 = self.up2(d3)

        d2 = torch.cat(
            [
                d2,
                e2
            ],
            dim=1
        )

        d2 = self.dec2(d2)


        # ----------------------------------------------------

        d1 = self.up1(d2)

        d1 = torch.cat(
            [
                d1,
                e1
            ],
            dim=1
        )

        d1 = self.dec1(d1)


        # ====================================================
        # RGB OUTPUT
        # ====================================================

        output = self.final(
            d1
        )

        return torch.sigmoid(
            output
        )


# ============================================================
# MAIN
# ============================================================

print()

print("=" * 70)
print("INFRARED → RGB IMAGE COLORIZATION")
print("=" * 70)

print(
    f"Device        : {DEVICE}"
)

print(
    f"Image size    : {IMAGE_SIZE}"
)

print(
    f"Batch size    : {BATCH_SIZE}"
)

print(
    f"Epochs        : {NUM_EPOCHS}"
)

print(
    f"Learning rate : {LEARNING_RATE}"
)

print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print()

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)


train_dataset = ColorizationDataset(
    TRAIN_IR_DIR,
    TRAIN_RGB_DIR
)

val_dataset = ColorizationDataset(
    VAL_IR_DIR,
    VAL_RGB_DIR
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


print()

print(
    f"Training samples  : {len(train_dataset)}"
)

print(
    f"Validation samples: {len(val_dataset)}"
)


# ============================================================
# CREATE MODEL
# ============================================================

print()

print("=" * 70)
print("CREATING LIGHTWEIGHT U-NET")
print("=" * 70)


model = UNet().to(
    DEVICE
)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    f"Total parameters     : {total_parameters:,}"
)

print(
    f"Trainable parameters : {trainable_parameters:,}"
)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.L1Loss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# RESUME TRAINING FROM EPOCH 18
# ============================================================

# You have already completed Epoch 17.
# Therefore, training will continue from Epoch 18.

START_EPOCH = 17

# Best validation loss obtained during previous training.
# Based on your earlier training results.
best_val_loss = 0.240712


print()

print("=" * 70)
print("RESUMING TRAINING")
print("=" * 70)

print(
    f"Completed epochs     : {START_EPOCH}"
)

print(
    f"Resuming from        : Epoch {START_EPOCH + 1}"
)

print(
    f"Final epoch          : {NUM_EPOCHS}"
)

print(
    f"Best validation loss : {best_val_loss:.6f}"
)


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print()

    print("ERROR: Model checkpoint not found!")

    print(
        f"Expected model at: {MODEL_PATH}"
    )

    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )


print()

print(
    "Loading existing model checkpoint..."
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ------------------------------------------------------------
# Load model weights
# ------------------------------------------------------------

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

print(
    "✓ Model weights loaded"
)


# ------------------------------------------------------------
# Load optimizer state if available
# ------------------------------------------------------------

if "optimizer_state_dict" in checkpoint:

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    print(
        "✓ Optimizer state loaded"
    )

else:

    print(
        "⚠ Optimizer state not found."
    )


print()

print(
    "Training will continue from Epoch 18."
)

print("=" * 70)


# ============================================================
# TRAINING
# ============================================================

train_losses = []

val_losses = []


print()

print("=" * 70)
print("STARTING / RESUMING TRAINING")
print("=" * 70)


# ============================================================
# EPOCH LOOP
# ============================================================

for epoch in range(
    START_EPOCH,
    NUM_EPOCHS
):


    # ========================================================
    # TRAINING PHASE
    # ========================================================

    model.train()

    running_train_loss = 0.0


    for batch_idx, (ir, rgb) in enumerate(
        train_loader
    ):

        ir = ir.to(
            DEVICE
        )

        rgb = rgb.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = model(
            ir
        )


        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            output,
            rgb
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()


        running_train_loss += (
            loss.item()
        )


        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % 50 == 0
            or
            (batch_idx + 1) == len(train_loader)
        ):

            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.6f}"
            )


    # ========================================================
    # AVERAGE TRAINING LOSS
    # ========================================================

    train_loss = (
        running_train_loss
        /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    running_val_loss = 0.0


    with torch.no_grad():

        for ir, rgb in val_loader:

            ir = ir.to(
                DEVICE
            )

            rgb = rgb.to(
                DEVICE
            )


            output = model(
                ir
            )


            loss = criterion(
                output,
                rgb
            )


            running_val_loss += (
                loss.item()
            )


    # ========================================================
    # AVERAGE VALIDATION LOSS
    # ========================================================

    val_loss = (
        running_val_loss
        /
        len(val_loader)
    )


    # ========================================================
    # STORE LOSSES
    # ========================================================

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print()

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} completed"
    )

    print(
        f"Train Loss      : {train_loss:.6f}"
    )

    print(
        f"Validation Loss : {val_loss:.6f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss
            },
            MODEL_PATH
        )


        print(
            "✓ Best model saved"
        )

    else:

        print(
            "Best model not improved"
        )


    print(
        "-" * 70
    )


# ============================================================
# SAVE LOSS GRAPH
# ============================================================

if len(train_losses) > 0:

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        range(
            START_EPOCH + 1,
            NUM_EPOCHS + 1
        ),
        train_losses,
        label="Training Loss"
    )


    plt.plot(
        range(
            START_EPOCH + 1,
            NUM_EPOCHS + 1
        ),
        val_losses,
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "L1 Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()


    plt.savefig(
        LOSS_PLOT_PATH,
        dpi=150
    )

    plt.close()


# ============================================================
# FINISHED
# ============================================================

print()

print("=" * 70)
print("TRAINING SESSION COMPLETED")
print("=" * 70)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Model checkpoint: "
    f"{MODEL_PATH}"
)

print(
    f"Loss graph: "
    f"{LOSS_PLOT_PATH}"
)

print("=" * 70)