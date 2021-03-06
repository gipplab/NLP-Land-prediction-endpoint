"""Setup for the module."""
# -*- coding: utf-8 -*-
from setuptools import setup  # type: ignore

packages = [
    "nlp_land_prediction_endpoint",
    "nlp_land_prediction_endpoint.enums",
    "nlp_land_prediction_endpoint.models",
    "nlp_land_prediction_endpoint.routes",
]

package_data = {"": ["*"]}

install_requires = [
    "bcrypt>=3.2.0,<4.0.0",
    "bson>=0.5.10,<0.6.0",
    "fastapi>=0.70.0,<0.71.0",
    "pydantic>=1.8.2,<2.0.0",
    "uvicorn>=0.15.0,<0.16.0",
]

setup_kwargs = {
    "name": "nlp-land-prediction-endpoint",
    "version": "0.1.0",
    "description": (
        "This repository implements the processing of machine-learning methods for"
        " NLP-Land-backend."
    ),
    "long_description": None,
    "author": "Jan Philip Wahle",
    "author_email": "wahle@uni-wuppertal.de",
    "maintainer": "Jan Philip Wahle",
    "maintainer_email": "wahle@uni-wuppertal.de",
    "url": "https://github.com/ag-gipp/NLP-Land-prediction-endpoint",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.8,<3.10",
}


setup(**setup_kwargs)
