"""
HTTP serving of a MILIA deployment package (one FastAPI app for every serving target).

The app loads a verified package (:func:`.package.verify_package`) once at startup with
``Predictor.from_checkpoint``. It featurizes each SMILES with the training run's
``structural_features_config`` and applies the training data's recorded pre-transforms, exactly
as ``main.py --predict`` does, and returns one row of model outputs per molecule. Input is
always parsed as SMILES (``format="smiles"``): automatic format detection would let a request
string name a file on the server.

Routes, one app for every target:

* ``GET /ping``, ``POST /invocations``: Amazon SageMaker hosting (port 8080; SageMaker starts
  the container with the argument ``serve``, which :func:`main` accepts).
* ``AIP_HEALTH_ROUTE`` / ``AIP_PREDICT_ROUTE`` on ``AIP_HTTP_PORT``: Vertex AI custom
  containers, when Vertex sets these variables.
* ``GET /health``, ``POST /predict``: Azure ML, Kubernetes probes and plain Docker.

Request body (the Vertex AI shape, used for every predict route)::

    {"instances": ["CCO", "c1ccccc1"]}

Response::

    {"predictions": [[...], [...]], "model_name": "...", "task_type": "...",
     "output_names": ["...", ...]}

At startup the app featurizes the package's input example and checks it against the package's
signature (:meth:`PackagedPredictor.check_signature`), so a MILIA that featurizes differently
from the one that trained the model refuses to serve it.

Run: ``python -m milia_pipeline.models.deployment.serving --package /opt/ml/model``.
Requires the ``serve`` extra (FastAPI, Uvicorn), which MILIA's Docker images install.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from milia_pipeline.exceptions import DeploymentError
from milia_pipeline.models.deployment.drift import DriftMonitor, drift_settings
from milia_pipeline.models.deployment.package import MAX_SMILES_LENGTH, verify_package

__all__ = [
    "DEFAULT_PACKAGE_DIR",
    "PackagedPredictor",
    "PredictRequest",
    "PredictResponse",
    "create_app",
    "default_base_image",
    "main",
    "write_serving_dockerfile",
]

logger = logging.getLogger(__name__)

DEFAULT_PACKAGE_DIR = "/opt/ml/model"  # SageMaker's model directory; the image copies packages here
DEFAULT_PORT = 8080  # SageMaker requires 8080; Vertex AI's AIP_HTTP_PORT defaults to it
DEFAULT_MAX_INSTANCES = 1000  # MILIA_SERVE_MAX_INSTANCES overrides it


class PredictRequest(BaseModel):
    """Request body of every predict route (the Vertex AI shape); unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")
    instances: list[str] = Field(min_length=1)
    parameters: dict[str, Any] | None = None


class PredictResponse(BaseModel):
    """One row of model outputs per instance, in request order.

    ``output_names`` names the columns, when the package's signature records them.
    ``intervals`` (when the package has a conformal calibration): ``alpha``, ``method`` and
    ``lower``/``upper`` rows, each covering the true value with probability at least
    ``1 - alpha`` per target for molecules like the calibration set.
    """

    predictions: list[list[float]]
    model_name: str
    task_type: str
    output_names: list[str] | None = None
    intervals: dict[str, Any] | None = None


class PackagedPredictor:
    """A deployment package's model, ready to predict from SMILES."""

    def __init__(self, predictor: Any, manifest: dict[str, Any]):
        self.predictor = predictor
        self.manifest = manifest
        # Set by from_package (deployment/drift.py); None: requests are not monitored
        self.drift: DriftMonitor | None = None

    @classmethod
    def from_package(
        cls, package_dir: str | Path, device: torch.device | None = None
    ) -> PackagedPredictor:
        """Verify the package (format, SHA-256), then load its model with ``Predictor``."""
        from milia_pipeline.models.post_training.inference.predictor import Predictor

        checkpoint, manifest = verify_package(package_dir)
        predictor = Predictor.from_checkpoint(
            checkpoint_path=checkpoint, working_root_dir=Path(package_dir), device=device
        )
        packaged = cls(predictor, manifest)
        packaged.check_signature()
        # Requests compared with the training inputs (deployment/drift.py); after the signature
        # check, so only served molecules enter the window
        threshold, window = drift_settings()
        packaged.drift = DriftMonitor(manifest.get("input_reference"), threshold, window)
        return packaged

    @property
    def output_names(self) -> list[str] | None:
        """The output column names of the package's signature (``None``: not recorded)."""
        names = ((self.manifest.get("signature") or {}).get("outputs") or {}).get("names")
        return list(names) if names else None

    def check_signature(self) -> None:
        """Featurize a molecule as requests are featurized and compare it with the signature.

        The model was trained on ``signature.inputs.node_features`` columns; MILIA code that
        featurizes differently would compute wrong values rather than fail, so the server refuses
        to start instead. A version-1 package has no signature and is not checked.

        Raises:
            DeploymentError: The featurized width or the output count differs from the signature.
        """
        signature = self.manifest.get("signature")
        if not signature:
            return
        expected_inputs = (signature.get("inputs") or {}).get("node_features")
        expected_outputs = (signature.get("outputs") or {}).get("count")
        smiles = (self.manifest.get("input_example") or ["C"])[0]
        (graph,) = self.featurize([smiles])
        width = graph.x.size(-1) if getattr(graph, "x", None) is not None else None
        if expected_inputs is not None and width != expected_inputs:
            raise DeploymentError(
                f"This MILIA featurizes {smiles!r} into {width} node features; the package's "
                f"model was trained on {expected_inputs}. Serve it with the MILIA version that "
                f"wrote it ({self.manifest.get('milia_version')})."
            )
        if expected_outputs is not None:
            out = self.predictor.predict_batch([graph], return_numpy=True).reshape(1, -1)
            if out.shape[1] != expected_outputs:
                raise DeploymentError(
                    f"The model returns {out.shape[1]} values per molecule; the package's "
                    f"signature records {expected_outputs}"
                )

    def featurize(self, smiles: Sequence[str]) -> list[Any]:
        """Graphs for SMILES, prepared as the training graphs were.

        Featurized with the training run's ``structural_features_config``, then passed through
        the training data's pre-transforms (``Predictor.prepare_inputs``).

        Raises:
            ValueError: A SMILES cannot be parsed; the message names its index.
        """
        from milia_pipeline.models.post_training import convert_to_pyg

        config = self.predictor.structural_features_config
        graphs = []
        for index, text in enumerate(smiles):
            try:
                graphs.append(
                    convert_to_pyg(text, format="smiles", structural_features_config=config)
                )
            except ValueError as exc:
                raise ValueError(f"instances[{index}]: {exc}") from exc
        graphs = self.predictor.prepare_inputs(graphs)
        if self.drift is not None:
            self.drift.observe(graphs)
        return graphs

    def predict_smiles(self, smiles: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """One row of model outputs per SMILES, in input order.

        Raises:
            ValueError: A SMILES cannot be parsed; the message names its index.
        """
        graphs = self.featurize(smiles)
        predictions = self.predictor.predict_batch(graphs, batch_size=batch_size, return_numpy=True)
        return predictions.reshape(len(graphs), -1).tolist()

    def predict_smiles_with_intervals(
        self, smiles: Sequence[str], batch_size: int = 32
    ) -> tuple[list[list[float]], dict[str, Any] | None]:
        """Rows as :meth:`predict_smiles`, plus conformal intervals when the package has them."""
        if not self.predictor.calibration:
            return self.predict_smiles(smiles, batch_size), None
        graphs = self.featurize(smiles)
        result = self.predictor.predict_with_intervals(graphs, batch_size=batch_size)
        intervals = {
            "alpha": result["alpha"],
            "method": result["method"],
            "lower": result["lower"].tolist(),
            "upper": result["upper"].tolist(),
        }
        return result["mean"].tolist(), intervals


def _max_instances() -> int:
    raw = os.environ.get("MILIA_SERVE_MAX_INSTANCES", str(DEFAULT_MAX_INSTANCES))
    try:
        value = int(raw)
    except ValueError as exc:
        raise DeploymentError(f"MILIA_SERVE_MAX_INSTANCES must be an integer, got {raw!r}") from exc
    if value < 1:
        raise DeploymentError(f"MILIA_SERVE_MAX_INSTANCES must be at least 1, got {value}")
    return value


def create_app(package_dir: str | Path, device: torch.device | None = None) -> Any:
    """The FastAPI app serving ``package_dir``. The model loads at startup (fail fast)."""
    from fastapi import FastAPI, HTTPException
    from fastapi.concurrency import run_in_threadpool

    max_instances = _max_instances()
    state: dict[str, PackagedPredictor] = {}

    @asynccontextmanager
    async def lifespan(_app):
        state["model"] = PackagedPredictor.from_package(package_dir, device=device)
        logger.info(f"Serving {state['model'].manifest['model_name']} from {package_dir}")
        yield
        state.clear()

    app = FastAPI(title="MILIA model server", lifespan=lifespan)

    def health() -> dict[str, str]:
        if "model" not in state:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "healthy"}

    async def predict(request: PredictRequest) -> PredictResponse:
        model = state["model"]
        if len(request.instances) > max_instances:
            raise HTTPException(
                status_code=422,
                detail=f"{len(request.instances)} instances; at most {max_instances} per request",
            )
        too_long = [i for i, s in enumerate(request.instances) if len(s) > MAX_SMILES_LENGTH]
        if too_long:
            raise HTTPException(
                status_code=422,
                detail=f"instances {too_long} exceed {MAX_SMILES_LENGTH} characters",
            )
        try:
            # Featurization and the forward pass are CPU-bound: run them off the event loop.
            rows, intervals = await run_in_threadpool(
                model.predict_smiles_with_intervals, request.instances
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictResponse(
            predictions=rows,
            model_name=model.manifest["model_name"],
            task_type=model.manifest["task_type"],
            output_names=model.output_names,
            intervals=intervals,
        )

    def drift() -> dict[str, Any]:
        monitor = state["model"].drift
        return monitor.report() if monitor is not None else DriftMonitor(None).report()

    app.add_api_route("/monitoring/drift", drift, methods=["GET"])

    health_routes = ["/health", "/ping"]
    predict_routes = ["/predict", "/invocations"]
    for variable, routes in (
        ("AIP_HEALTH_ROUTE", health_routes),
        ("AIP_PREDICT_ROUTE", predict_routes),
    ):
        route = os.environ.get(variable)
        if route and route not in routes:
            routes.append(route)
    for route in health_routes:
        app.add_api_route(route, health, methods=["GET"])
    for route in predict_routes:
        app.add_api_route(route, predict, methods=["POST"], response_model=PredictResponse)
    return app


IMAGE_REPOSITORY = "ghcr.io/shahram-boshra/milia"

_DOCKERFILE = """\
# Generated by MILIA {version}: serves the deployment package in this directory.
# Build: docker build -t milia-serve .   Run: docker run -p 8080:8080 milia-serve
# The base image must carry MILIA's `serve` extra (FastAPI, Uvicorn); the build checks it and the
# package's SHA-256 before the image is usable.
FROM {base_image}
{user_line}COPY deployment.json model.pt {package_dir}/
ENV MILIA_PACKAGE_DIR={package_dir}
RUN python -c "import fastapi, uvicorn; \\
from milia_pipeline.models.deployment.package import verify_package; \\
verify_package('{package_dir}')"
EXPOSE {port}
ENTRYPOINT ["python", "-m", "milia_pipeline.models.deployment.serving"]
"""


def default_base_image() -> str:
    """The published MILIA image of the version that writes the package (same code serves it)."""
    from milia_pipeline import __version__

    return f"{IMAGE_REPOSITORY}:{__version__}"


def write_serving_dockerfile(
    package_dir: str | Path, base_image: str | None = None, run_as_root: bool = False
) -> Path:
    """Write a Dockerfile next to a package that serves it with :func:`main`.

    Args:
        package_dir: A deployment package (``model.pt`` + ``deployment.json``).
        base_image: Image to build on; defaults to :func:`default_base_image`.
        run_as_root: Run the server as root. Amazon SageMaker expects containers to run as root;
            elsewhere the base image's unprivileged ``milia`` user is kept.

    Returns:
        The Dockerfile path.
    """
    from milia_pipeline import __version__

    verify_package(package_dir)
    dockerfile = Path(package_dir) / "Dockerfile"
    dockerfile.write_text(
        _DOCKERFILE.format(
            version=__version__,
            base_image=base_image or default_base_image(),
            user_line="USER root\n" if run_as_root else "",
            package_dir=DEFAULT_PACKAGE_DIR,
            port=DEFAULT_PORT,
        )
    )
    return dockerfile


def main(argv: Sequence[str] | None = None) -> int:
    """Serve a package with Uvicorn (one process per container; scale with replicas)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("command", nargs="?", choices=["serve"], help="accepted for SageMaker")
    parser.add_argument(
        "--package", default=os.environ.get("MILIA_PACKAGE_DIR", DEFAULT_PACKAGE_DIR)
    )
    # A server inside a container listens on all interfaces; the container exposes one port.
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("AIP_HTTP_PORT", DEFAULT_PORT))
    )
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(create_app(args.package), host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
