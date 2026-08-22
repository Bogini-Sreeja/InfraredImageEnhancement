from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import shutil

from preprocessing.preprocess import preprocess_image
from preprocessing.enhance import enhance_image
from preprocessing.super_resolution import super_resolve_image
from preprocessing.colorization import colorize_image


# =========================================
# CREATE FASTAPI APP
# =========================================

app = FastAPI(
    title="Infrared Image Enhancement API"
)


# =========================================
# CORS
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# DIRECTORIES
# =========================================

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================
# SERVE OUTPUT IMAGES
# =========================================

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)


# =========================================
# HOME
# =========================================

@app.get("/")
def home():

    return {
        "message": "Infrared Image Enhancement API is running"
    }


# =========================================
# STEP 1 — IMAGE PREPROCESSING
# =========================================

@app.post("/preprocess")
async def preprocess(
    file: UploadFile = File(...)
):

    # Save uploaded image
    input_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(
        input_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Create processed filename
    processed_filename = (
        f"processed_{file.filename}"
    )

    # Output path
    output_path = os.path.join(
        OUTPUT_DIR,
        processed_filename
    )

    # Run preprocessing
    preprocess_image(
        input_path,
        output_path
    )

    return {
        "message":
            "Image preprocessing completed",

        "filename":
            processed_filename,

        "url":
            f"/outputs/{processed_filename}"
    }


# =========================================
# STEP 2 — IMAGE ENHANCEMENT
# =========================================

@app.post("/enhance")
async def enhance(
    filename: str
):

    # Path of preprocessed image
    input_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    # Check whether image exists
    if not os.path.exists(input_path):

        return {
            "error":
                "Preprocessed image not found"
        }

    # Enhanced filename
    enhanced_filename = (
        f"enhanced_{filename}"
    )

    # Output path
    output_path = os.path.join(
        OUTPUT_DIR,
        enhanced_filename
    )

    # Run enhancement
    enhance_image(
        input_path,
        output_path
    )

    return {
        "message":
            "Image enhancement completed",

        "filename":
            enhanced_filename,

        "url":
            f"/outputs/{enhanced_filename}"
    }


# =========================================
# STEP 3 — SUPER RESOLUTION
# =========================================

@app.post("/super-resolution")
async def super_resolution(
    filename: str
):

    # Path of enhanced image
    input_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    # Check whether image exists
    if not os.path.exists(input_path):

        return {
            "error":
                "Enhanced image not found"
        }

    # Super-resolution filename
    sr_filename = (
        f"sr_{filename}"
    )

    # Output path
    output_path = os.path.join(
        OUTPUT_DIR,
        sr_filename
    )

    # Run super-resolution
    super_resolve_image(
        input_path,
        output_path
    )

    return {
        "message":
            "Super-resolution completed",

        "filename":
            sr_filename,

        "url":
            f"/outputs/{sr_filename}"
    }
# =========================================
# STEP 4 — RGB COLORIZATION
# =========================================

@app.post("/colorize")
async def colorize(
    filename: str
):

    # Path of super-resolved image
    input_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    # Check whether image exists
    if not os.path.exists(input_path):

        return {
            "error":
                "Super-resolved image not found"
        }

    # Colorized filename
    colorized_filename = (
        f"colorized_{filename}"
    )

    # Output path
    output_path = os.path.join(
        OUTPUT_DIR,
        colorized_filename
    )

    # Run colorization
    colorize_image(
        input_path,
        output_path
    )

    return {
        "message":
            "RGB colorization completed",

        "filename":
            colorized_filename,

        "url":
            f"/outputs/{colorized_filename}"
    }