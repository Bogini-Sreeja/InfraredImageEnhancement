from preprocessing.colorization import colorize_image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import shutil

from preprocessing.preprocess import preprocess_image
from preprocessing.enhance import enhance_image
from preprocessing.super_resolution import super_resolve_image


# --------------------------------------------------
# BASE DIRECTORIES
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Infrared Image Enhancement API",
    description="API for infrared image preprocessing, enhancement and super-resolution",
    version="1.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# STATIC OUTPUT FILES
# --------------------------------------------------

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Infrared Image Enhancement API is running"
    }


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):

    input_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    processed_filename = f"processed_{file.filename}"

    output_path = os.path.join(
        OUTPUT_DIR,
        processed_filename
    )

    preprocess_image(
        input_path,
        output_path
    )

    return {
        "message": "Image preprocessing completed",
        "filename": processed_filename,
        "url": f"/outputs/{processed_filename}"
    }


# --------------------------------------------------
# ENHANCEMENT
# --------------------------------------------------

@app.post("/enhance")
async def enhance(filename: str):

    input_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if not os.path.exists(input_path):
        return {
            "error": "Preprocessed image not found"
        }

    enhanced_filename = f"enhanced_{filename}"

    output_path = os.path.join(
        OUTPUT_DIR,
        enhanced_filename
    )

    enhance_image(
        input_path,
        output_path
    )

    return {
        "message": "Image enhancement completed",
        "filename": enhanced_filename,
        "url": f"/outputs/{enhanced_filename}"
    }


# --------------------------------------------------
# SUPER RESOLUTION
# --------------------------------------------------

@app.post("/super-resolution")
async def super_resolution(filename: str):

    input_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if not os.path.exists(input_path):
        return {
            "error": "Enhanced image not found"
        }

    sr_filename = f"sr_{filename}"

    output_path = os.path.join(
        OUTPUT_DIR,
        sr_filename
    )

    super_resolve_image(
        input_path,
        output_path
    )

    return {
        "message": "Super-resolution completed",
        "filename": sr_filename,
        "url": f"/outputs/{sr_filename}"
    }
# --------------------------------------------------
# RGB COLORIZATION
# --------------------------------------------------

@app.post("/colorize")
async def colorize(filename: str):

    input_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if not os.path.exists(input_path):
        return {
            "error": "Super-resolved image not found"
        }

    colorized_filename = f"colorized_{filename}"

    output_path = os.path.join(
        OUTPUT_DIR,
        colorized_filename
    )

    colorize_image(
        input_path,
        output_path
    )

    return {
        "message": "RGB colorization completed",
        "filename": colorized_filename,
        "url": f"/outputs/{colorized_filename}"
    }