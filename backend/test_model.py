import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "colorization_model.pth"
)

TEST_IR_DIR = os.path.join(
    BASE_DIR,
    "..",
    "dataset_split",
    "test",
    "infrared"
)

TEST_RGB_DIR = os.path.join(
    BASE_DIR,
    "..",
    "dataset_split",
    "test",
    "rgb"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "test_results"
)

PREDICTION_DIR = os.path.join(
    RESULT_DIR,
    "predictions"
)

COMPARISON_DIR = os.path.join(
    RESULT_DIR,
    "comparisons"
)


os.makedirs(
    PREDICTION_DIR,
    exist_ok=True
)

os.makedirs(
    COMPARISON_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DOUBLE CONVOLUTION BLOCK
# SAME AS TRAINING MODEL
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
# EXACT SAME ARCHITECTURE USED FOR TRAINING
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


        d2 = self.up2(d3)

        d2 = torch.cat(
            [
                d2,
                e2
            ],
            dim=1
        )

        d2 = self.dec2(d2)


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

        output = self.final(d1)

        return torch.sigmoid(
            output
        )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_test_image(
    ir_path,
    rgb_path
):

    # --------------------------------------------------------
    # Load IR
    # --------------------------------------------------------

    ir = Image.open(
        ir_path
    ).convert("L")


    # --------------------------------------------------------
    # Load RGB ground truth
    # --------------------------------------------------------

    rgb = Image.open(
        rgb_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Resize
    # SAME AS TRAINING
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Convert to NumPy
    # SAME NORMALIZATION AS TRAINING
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Convert IR to Tensor
    # H x W → 1 x H x W
    # --------------------------------------------------------

    ir = torch.from_numpy(
        ir
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Convert RGB to NumPy H x W x 3
    # --------------------------------------------------------

    return ir, rgb


# ============================================================
# CREATE SIDE-BY-SIDE COMPARISON
# ============================================================

def create_comparison(
    ir_array,
    prediction,
    ground_truth,
    output_path
):

    # --------------------------------------------------------
    # IR: grayscale → RGB
    # --------------------------------------------------------

    ir_uint8 = (
        ir_array * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    ir_rgb = np.stack(
        [
            ir_uint8,
            ir_uint8,
            ir_uint8
        ],
        axis=2
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_uint8 = (
        prediction * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth_uint8 = (
        ground_truth * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # Combine:
    #
    # IR | Predicted RGB | Ground Truth RGB
    # --------------------------------------------------------

    comparison = np.concatenate(
        [
            ir_rgb,
            prediction_uint8,
            ground_truth_uint8
        ],
        axis=1
    )


    comparison_image = Image.fromarray(
        comparison
    )

    comparison_image.save(
        output_path
    )


# ============================================================
# MAIN TESTING FUNCTION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("INFRARED IMAGE COLORIZATION - MODEL TESTING")
    print("=" * 70)

    print()

    print(
        "Device:",
        DEVICE
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Test IR folder:",
        TEST_IR_DIR
    )

    print(
        "Test RGB folder:",
        TEST_RGB_DIR
    )


    # ========================================================
    # CHECK PATHS
    # ========================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )


    if not os.path.exists(
        TEST_IR_DIR
    ):

        raise FileNotFoundError(
            f"Test IR folder not found:\n{TEST_IR_DIR}"
        )


    if not os.path.exists(
        TEST_RGB_DIR
    ):

        raise FileNotFoundError(
            f"Test RGB folder not found:\n{TEST_RGB_DIR}"
        )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    print()
    print("-" * 70)
    print("LOADING TRAINED MODEL")
    print("-" * 70)


    model = UNet().to(
        DEVICE
    )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    # Your train.py saves a checkpoint dictionary
    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        print(
            "Checkpoint epoch:",
            checkpoint.get(
                "epoch",
                "Unknown"
            )
        )


        print(
            "Best validation loss:",
            checkpoint.get(
                "val_loss",
                "Unknown"
            )
        )


    else:

        # Handle direct state_dict just in case
        model.load_state_dict(
            checkpoint
        )


    model.eval()


    print(
        "Model loaded successfully."
    )


    # ========================================================
    # GET TEST FILES
    # ========================================================

    ir_files = sorted(
        [
            f
            for f in os.listdir(
                TEST_IR_DIR
            )
            if f.lower().endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg"
                )
            )
        ]
    )


    rgb_files = sorted(
        [
            f
            for f in os.listdir(
                TEST_RGB_DIR
            )
            if f.lower().endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg"
                )
            )
        ]
    )


    print()
    print("-" * 70)
    print("TEST DATASET")
    print("-" * 70)


    print(
        "IR images :",
        len(ir_files)
    )

    print(
        "RGB images:",
        len(rgb_files)
    )


    if len(ir_files) == 0:

        raise ValueError(
            "No IR test images found."
        )


    if len(rgb_files) == 0:

        raise ValueError(
            "No RGB test images found."
        )


    # ========================================================
    # CHECK MATCHING FILENAMES
    # ========================================================

    ir_set = set(ir_files)

    rgb_set = set(rgb_files)


    missing_rgb = sorted(
        ir_set - rgb_set
    )

    missing_ir = sorted(
        rgb_set - ir_set
    )


    if missing_rgb:

        print()
        print(
            "Missing RGB files:"
        )

        for f in missing_rgb:

            print(
                f
            )


        raise ValueError(
            "Some IR files do not have matching RGB files."
        )


    if missing_ir:

        print()
        print(
            "Missing IR files:"
        )

        for f in missing_ir:

            print(
                f
            )


        raise ValueError(
            "Some RGB files do not have matching IR files."
        )


    # ========================================================
    # TESTING
    # ========================================================

    psnr_values = []

    ssim_values = []


    print()
    print("-" * 70)
    print("STARTING MODEL TESTING")
    print("-" * 70)


    with torch.no_grad():

        for index, filename in enumerate(
            ir_files
        ):

            # ------------------------------------------------
            # Paths
            # ------------------------------------------------

            ir_path = os.path.join(
                TEST_IR_DIR,
                filename
            )

            rgb_path = os.path.join(
                TEST_RGB_DIR,
                filename
            )


            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            ir_tensor, ground_truth = load_test_image(
                ir_path,
                rgb_path
            )


            # ------------------------------------------------
            # Add batch dimension
            #
            # 1 x H x W
            # →
            # 1 x 1 x H x W
            # ------------------------------------------------

            ir_tensor = ir_tensor.unsqueeze(
                0
            )


            ir_tensor = ir_tensor.to(
                DEVICE
            )


            # ------------------------------------------------
            # Generate prediction
            # ------------------------------------------------

            prediction = model(
                ir_tensor
            )


            # ------------------------------------------------
            # Remove batch dimension
            #
            # 1 x 3 x H x W
            # →
            # 3 x H x W
            # ------------------------------------------------

            prediction = prediction.squeeze(
                0
            ).cpu().numpy()


            # ------------------------------------------------
            # Convert:
            #
            # 3 x H x W
            # →
            # H x W x 3
            # ------------------------------------------------

            prediction = np.transpose(
                prediction,
                (
                    1,
                    2,
                    0
                )
            )


            prediction = np.clip(
                prediction,
                0,
                1
            )


            # ------------------------------------------------
            # PSNR
            # ------------------------------------------------

            psnr = peak_signal_noise_ratio(
                ground_truth,
                prediction,
                data_range=1.0
            )


            # ------------------------------------------------
            # SSIM
            # ------------------------------------------------

            ssim = structural_similarity(
                ground_truth,
                prediction,
                channel_axis=2,
                data_range=1.0
            )


            psnr_values.append(
                psnr
            )

            ssim_values.append(
                ssim
            )


            # ------------------------------------------------
            # Save prediction
            # ------------------------------------------------

            prediction_uint8 = (
                prediction * 255
            ).clip(
                0,
                255
            ).astype(
                np.uint8
            )


            prediction_image = Image.fromarray(
                prediction_uint8
            )


            prediction_path = os.path.join(
                PREDICTION_DIR,
                filename
            )


            prediction_image.save(
                prediction_path
            )


            # ------------------------------------------------
            # Save comparison
            # ------------------------------------------------

            comparison_path = os.path.join(
                COMPARISON_DIR,
                filename
            )


            ir_array = ir_tensor.squeeze(
                0
            ).squeeze(
                0
            ).cpu().numpy()


            create_comparison(
                ir_array,
                prediction,
                ground_truth,
                comparison_path
            )


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            print(
                f"[{index + 1:03d}/{len(ir_files):03d}] "
                f"{filename} | "
                f"PSNR: {psnr:.4f} dB | "
                f"SSIM: {ssim:.4f}"
            )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    average_psnr = np.mean(
        psnr_values
    )


    average_ssim = np.mean(
        ssim_values
    )


    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)


    print(
        f"Test images           : {len(ir_files)}"
    )

    print(
        f"Average PSNR          : {average_psnr:.4f} dB"
    )

    print(
        f"Average SSIM          : {average_ssim:.4f}"
    )


    print()
    print(
        "Predictions saved to:"
    )

    print(
        PREDICTION_DIR
    )


    print()
    print(
        "Comparisons saved to:"
    )

    print(
        COMPARISON_DIR
    )


    print()
    print("=" * 70)
    print("TESTING COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()