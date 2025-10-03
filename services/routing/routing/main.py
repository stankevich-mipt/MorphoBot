#    Copyright 2025, Stankevich Andrey, stankevich.as@phystech.edu

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""FastAPI microservice for face detection, alignment, and gender classification."""

from contextlib import asynccontextmanager
from dataclasses import asdict
import logging
import os
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse

from .inference import (
    ClassificationResult,
    create_image_processor,
    InferenceError
)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
ARTIFACTS_OUTPUT_DIR = Path(
    os.getenv("ARTIFACTS_OUTPUT_DIR", "./artifacts")
)


@asynccontextmanager   # type: ignore
async def lifespan(app: FastAPI):
    """App lifespan manager for model loading and cleanup."""
    try:
        app.state.processor = create_image_processor(
            classifier_alias=MODEL_ALIAS,
            output_path=ARTIFACTS_OUTPUT_DIR,
        )
        logger.info("Models loaded successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
    finally:
        logger.info("Shutting down Face Classification Service.")


app = FastAPI(
    title="Face Classifier API",
    description=(
        "Microservice for face detection, "
        "alignment, and gender classification"
    ),
    version="0.1.0",
    lifespan=lifespan
)


@app.exception_handler(InferenceError)
async def inference_error_handler(
    request: Request, exc: InferenceError
):
    """Global exception handler for inference errors."""
    logger.error(f"Inference error on {request.url}: {exc}")
    return JSONResponse(
        status_code=exc.status,
        content={
            "error": exc.code,
            "detail": str(exc),
            "status": "error"
        }
    )


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "service": "Routing API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "classify": "/classify"
        }
    }


@app.get("/health")
def health_check():
    """Check health by accessing the processor attribute in app state."""
    if not hasattr(app.state, "processor") or app.state.processor is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Models not loaded"}
        )

    return {
        "status": "healthy",
        "service": "routing-api",
        "model_alias": MODEL_ALIAS
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Classify gender from uploaded face image.

    Args:
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        JSON response with classification results
    """
    if not hasattr(app.state, "processor") or app.state.processor is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Process image through the pipeline
        result: ClassificationResult = app.state.processor.classify_face(image_bytes)
        logger.info(
            f"Classification successful: {result.predicted_class}"
            f" ({result.confidence:.3f})"
        )
        return asdict(result)

    except InferenceError:
        raise

    except Exception as e:
        logger.error(f"Unexpected error during classification: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
