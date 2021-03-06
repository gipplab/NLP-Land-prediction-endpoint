"""This module implements the status endpoint."""
from typing import Dict

from fastapi import APIRouter

import nlp_land_prediction_endpoint

router = APIRouter()


@router.get("/", response_description="Status of the backend.")
async def read_root() -> Dict[str, str]:
    """Status endpoint of the API.

    Returns:
        Dict[str, str]: A message of the current status and version.
    """
    return {
        "message": (
            "NLP-Land-prediction-endpoint online at version "
            f"{nlp_land_prediction_endpoint.__version__}."
        )
    }
