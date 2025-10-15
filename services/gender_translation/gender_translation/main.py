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

"""FastAPI microservice for gender translation.

At the moment the only supported backend is CycleGAN, with
more models coming in the nearest future.
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile
)
from fastapi.responses import JSONResponse, Response

from .inference import create_translation_engine
from .schema import (
    ModelBackendType,
    TranslationError,
    TranslationResult,
)

def setup_logging(level=logging.INFO):  # noqa: D103
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(ch)


setup_logging(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)


MODEL_BACKEND = os.getenv("MODEL_BACKEND", "cyclegan")
MODEL_M2F_ALIAS = os.getenv("MODEL_M2F_ALIAS", "champion")
MODEL_F2M_ALIAS = os.getenv("MODEL_F2M_ALIAS", "champion")
ARTIFACTS_OUTPUT_DIR = os.getenv("ARTIFACTS_OUTPUT_DIR", "./artifacts")
USE_MIXED_PRECISION = os.getenv("USE_MIXED_PRECISION", "true").lower() == "true"
DEVICE = os.getenv("DEVICE", "auto")


@asynccontextmanager  # type: ignore
async def lifespan(app: FastAPI):
    """Lifespan is entered if translation engine is successfully created."""
    try:
        try:
            backend_type = ModelBackendType(MODEL_BACKEND.lower())
        except ValueError:
            raise ValueError(
                f"Unsupported backend: {MODEL_BACKEND}. "
                f"Supported: {[b.value for b in ModelBackendType]}"
            )

        engine = create_translation_engine(
            backend_type=backend_type,
            male_to_female_alias=MODEL_M2F_ALIAS,
            female_to_male_alias=MODEL_F2M_ALIAS,
            output_path=Path(ARTIFACTS_OUTPUT_DIR),
            device=DEVICE,
            use_mixed_precision=USE_MIXED_PRECISION
        )

        app.state.engine = engine

        logger.info(f"Gender Translation API started with {backend_type.value} backend.")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize Gender Translation API: {e}")
        raise

    finally:
        logger.info("Shutting down Gender Translation API")


app = FastAPI(
    title="Gender Translation API",
    description="""
    Framework-agnostic gender translation microservice with pluggable backends.

    ## Supported Backends
    - **CycleGAN**: Unpaired image-to-image translation

    ## Usage
    Upload an image and specify the source gender to get a translated image.
    The backend is configured via evironment variables.
    """,
    version="0.1.0",
    lifespan=lifespan
)


@app.exception_handler(TranslationError)
async def translation_error_handler(request: Request, exc: TranslationError):
    """Global exception hanlder for TranslationError exceptions."""
    return JSONResponse(
        status_code=exc.status,
        content={
            "error": exc.code,
            "detail": str(exc),
            "status": "error"
        }
    )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Gender Translation API",
        "version": "0.1.0",
        "description": "Framework-agnostic gender translation with pluggable backends",
        "current_backend": MODEL_BACKEND,
        "endpoints": {
            "heath": "/health",
            "translate": "/translate",
        },
        "supported_genders": ["male", "female"],
        "configuration": {
            "backend": MODEL_BACKEND,
            "device": DEVICE,
            "mixed_precision": USE_MIXED_PRECISION
        }
    }


@app.get("/health")
async def health_check():
    """Check health by calling health method of translation engine."""
    if not hasattr(app.state, "engine") or app.state.engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "gender-translation-api",
                "detail": "Translation engine not loaded"
            }
        )

    health_status = app.state.engine.health_check()

    if health_status["engine_ready"] and health_status["translation_model_ready"]:
        return {
            "status": "healthy",
            "service": "gender-translation-api",
            "backend": {
                "type": MODEL_BACKEND,
                "m2f_alias": MODEL_M2F_ALIAS,
                "f2m_alias": MODEL_F2M_ALIAS
            },
            "health": health_status
        }
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "partial",
                "service": "gender-translation-api",
                "detail": "Some components are not ready",
                "health": health_status
            }
        )


@app.post("/translate")
async def translate_gender(
    file: UploadFile = File(..., description="Image file containing a face"),
    source_gender: Literal["male", "female"] = Form(..., description="Source gender of the face"),
    include_debug: bool = Form(False, description="Include debug information in headers")
):
    """Translate gender in the provided face image using the configured backend.

    Args:
        file: Image file
        source_gender: Domain to translate from ("male", "female")
        include_debug: Add debug information into response headers.
    """
    if not hasattr(app.state, "engine") or app.state.engine is None:
        raise HTTPException(
            status_code=503, detail="Translation engine not ready")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image/*"
        )

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 Mb

    try:
        image_bytes = await file.read()
        file_size = len(image_bytes)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large: {file_size / 1024 / 1024:.1f}"
                    " Mb. Maximum: 10 Mb."
                )
            )

        engine_output = app.state.engine.translate_gender(
            image_bytes, source_gender
        )

        translated_bytes: bytes = engine_output[0]
        result: TranslationResult = engine_output[1]

        headers = {
            "X-Source-Gender": result.source_gender,
            "X-Target-Gender": result.target_gender,
            "X-Processing-Time-Ms": str(int(result.processing_time_ms)),
            "X-Face-Bbox": (
                f"{result.bbox[0]}, {result.bbox[1]}, "
                f"{result.bbox[2]}, {result.bbox[3]}"
            ),
            "X-Backend-Type": result.model_info.get("backend", MODEL_BACKEND),
            "Content-Type": "image/jpeg",
            "Content-Length": str(len(translated_bytes))
        }

        if include_debug:
            headers.update({
                "X-Model-Architecture": (
                    result.model_info.get("architecture", "unknown")
                ),
                "X-Device": (result.model_info.get("device", "unknown")),
                "X-Mixed-Precision": str(
                    result.model_info.get("mixed_precision", False)
                ),
                "X-Model-Parameters": str(
                    result.model_info.get("m2f_parameters", 0)
                    + result.model_info.get("f2m_parameters", 0)
                )
            })

        return Response(
            content=translated_bytes,
            media_type="image/jpeg",
            headers=headers
        )

    except TranslationError as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during translation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))

    logger.info(f"Starting server with {MODEL_BACKEND} backend...")
    logger.info(f"Available backends: {[b.value for b in ModelBackendType]}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )
