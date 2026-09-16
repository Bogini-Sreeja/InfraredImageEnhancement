import os

import cv2
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------
# DOUBLE CONVOLUTION BLOCK
# --------------------------------------------------

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
       return self.block(x)


# --------------------------------------------------
# U-NET
# --------------------------------------------------

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(64, 32)

        # RGB output
        self.final = nn.Conv2d(
            32,
            3,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        output = self.final(d1)

        return torch.sigmoid(output)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "colorization_model.pth"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNet().to(device)


# Load trained checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Your training script saved the model state
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)


model.eval()


# --------------------------------------------------
# COLORIZATION FUNCTION
# --------------------------------------------------

def colorize_image(input_path, output_path):

    # Read super-resolved grayscale image
    image = cv2.imread(
        input_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            "Could not read super-resolved image"
        )

    # Remember original SR dimensions
    original_height, original_width = image.shape

    # U-Net was trained using 256x256 images
    model_input = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    # Normalize to [0, 1]
    model_input = model_input.astype(
        np.float32
    ) / 255.0

    # Convert:
    # H,W -> 1,H,W -> 1,1,H,W
    tensor = torch.from_numpy(
        model_input
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(device)

    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

    with torch.no_grad():

        prediction = model(tensor)

    # Remove batch dimension
    prediction = prediction.squeeze(0)

    # Convert:
    # C,H,W -> H,W,C
    prediction = prediction.permute(
        1,
        2,
        0
    )

    # CPU + numpy
    prediction = prediction.cpu().numpy()

    # Convert [0,1] -> [0,255]
    prediction = np.clip(
        prediction * 255.0,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------
    # RESIZE RGB OUTPUT BACK TO SR SIZE
    # --------------------------------------------------

    colorized = cv2.resize(
        prediction,
        (original_width, original_height),
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------
    # SAVE RGB IMAGE
    # --------------------------------------------------

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            colorized,
            cv2.COLOR_RGB2BGR
        )
    )

    return output_path