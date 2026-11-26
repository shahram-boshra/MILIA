"""
HTTP serving of a MILIA deployment package (one FastAPI app for every serving target).

The app loads a verified package (:func:`.package.verify_package`) once at startup with
``Predictor.from_checkpoint``. An instance is a SMILES string (an identifier input, featurized
with the training run's ``structural_features_config``) or a structure object (atomic numbers,
coordinates in ångström, charge, identifier; built as the training dataset built its molecules,
``Predictor.prepare_structures``). Both pass the training filters and pre-transforms, exactly as
``main.py --predict`` does, and the app returns one row of model outputs per molecule. A string is
always parsed as SMILES (``format="smiles"``): automatic format detection would let a request
string name a file on the server.

Each instance kind is served when the model accepts it (``models/input_contract.py``) and, for
structures, when the package records a verified structure example; the app starts when it serves
at least one kind. A refused instance (a kind the model does not accept, outside the training
filters, a structure that cannot be built as the training molecules were) fails the request with
status 422, every refused instance named with its reason.

Routes, one app for every target:

* ``GET /ping``, ``POST /invocations``: Amazon SageMaker hosting (port 8080; SageMaker starts
  the container with the argument ``serve``, which :func:`main` accepts).
* ``AIP_HEALTH_ROUTE`` / ``AIP_PREDICT_ROUTE`` on ``AIP_HTTP_PORT``: Vertex AI custom
  containers, when Vertex sets these variables.
* ``GET /health``, ``POST /predict``: Azure ML, Kubernetes probes and plain Docker.

Request body (the Vertex AI shape, used for every predict route)::

    {"instances": ["CCO",
                   {"atomic_numbers": [8, 1, 1],
                    "coordinates": [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]],
                    "charge": 0, "identifier": "InChI=1S/H2O/h1H2", "identifier_type": "inchi"}]}

Response::

    {"predictions": [[...], [...]], "model_name": "...", "task_type": "...",
     "output_names": ["...", ...], "output_level": "graph"}

A graph-level model (task type ``graph_*``) answers one row of outputs per instance. A node-level
model (``node_*``) answers, per instance, one row per atom (``predictions[i][j]``: atom ``j`` of
instance ``i``), with ``"output_level": "node"`` and each instance's ``atomic_numbers`` in that
order. The app refuses to start for an edge-level model, or a task type naming no level
(:func:`output_level_refusal`).

At startup the app featurizes the package's examples (input example, structure example) and
checks them against the package's signature (:meth:`PackagedPredictor.check_signature`), so a
MILIA that featurizes differently from the one that trained the model refuses to serve it.

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
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from milia_pipeline.exceptions import DeploymentError, InputRefusedError
from milia_pipeline.models.capabilities import output_level
from milia_pipeline.models.deployment.drift import DriftMonitor, drift_settings
from milia_pipeline.models.deployment.package import MAX_SMILES_LENGTH, verify_package
from milia_pipeline.models.input_contract import IDENTIFIER, STRUCTURE
from milia_pipeline.models.post_training.conformal import json_bounds, unbounded_outputs

__all__ = [
    "DEFAULT_PACKAGE_DIR",
    "PackagedPredictor",
    "PredictRequest",
    "PredictResponse",
    "StructureInstance",
    "create_app",
    "default_base_image",
    "main",
    "require_servable",
    "write_serving_dockerfile",
]

logger = logging.getLogger(__name__)

DEFAULT_PACKAGE_DIR = "/opt/ml/model"  # SageMaker's model directory; the image copies packages here
DEFAULT_PORT = 8080  # SageMaker requires 8080; Vertex AI's AIP_HTTP_PORT defaults to it
DEFAULT_MAX_INSTANCES = 1000  # MILIA_SERVE_MAX_INSTANCES overrides it
MAX_STRUCTURE_ATOMS = (
    1000  # atoms of one structure instance (the request is bounded before it is built)
)


class StructureInstance(BaseModel):
    """A structure instance: a molecule with given coordinates (``StructureInput``'s fields).

    ``coordinates`` in ångström, one ``[x, y, z]`` per atomic number; ``charge``: the molecular
    charge (required by coordinate-based creation); ``identifier``/``identifier_type`` (``inchi``
    or ``smiles``): its identifier (required by identifier-based creation). Unknown fields are
    rejected.
    """

    model_config = ConfigDict(extra="forbid")
    atomic_numbers: list[int] = Field(min_length=1, max_length=MAX_STRUCTURE_ATOMS)
    coordinates: list[list[float]] = Field(min_length=1, max_length=MAX_STRUCTURE_ATOMS)
    charge: int | None = None
    identifier: str | None = Field(default=None, max_length=MAX_SMILES_LENGTH)
    identifier_type: Literal["inchi", "smiles"] | None = None
    name: str | None = Field(default=None, max_length=200)


class PredictRequest(BaseModel):
    """Request body of every predict route (the Vertex AI shape); unknown fields are rejected.

    Each instance is a SMILES string or a :class:`StructureInstance`.
    """

    model_config = ConfigDict(extra="forbid")
    instances: list[str | StructureInstance] = Field(min_length=1)
    parameters: dict[str, Any] | None = None


class PredictResponse(BaseModel):
    """Model outputs per instance, in request order.

    ``output_level`` ``"graph"``: one row of outputs per instance. ``"node"``: per instance, one
    row per atom, in the order of its ``atomic_numbers``. ``output_names`` names the columns, when
    the package's signature records them. ``intervals`` (when the package has a conformal
    calibration): ``alpha``, ``method`` and ``lower``/``upper`` shaped as ``predictions``, each
    covering the true value with probability at least ``1 - alpha`` per target for molecules like
    the calibration set (a node-level model's intervals cover all atoms of a molecule at once).
    An output whose calibration set was too small for ``1 - alpha`` has no finite bound: its
    bounds are ``null`` and ``intervals.unbounded`` names it.
    """

    predictions: list[list[float]] | list[list[list[float]]]
    model_name: str
    task_type: str
    output_names: list[str] | None = None
    output_level: Literal["graph", "node"] = "graph"
    atomic_numbers: list[list[int]] | None = None
    intervals: dict[str, Any] | None = None


def output_level_refusal(task_type: str | None) -> str | None:
    """Why the server cannot answer a model of ``task_type``, or ``None`` when it can.

    The task type's prefix names the output level
    (:func:`~milia_pipeline.models.capabilities.output_level`). A graph-level model gives one row
    per molecule and a node-level one one row per atom, which the response nests per instance
    (``PredictResponse``). An edge-level model gives one row per bond or atom pair, which the
    server does not map to its molecule; a task type without a known prefix names no level, so it
    is refused rather than guessed.
    """
    level = output_level(task_type)
    if level in ("graph", "node"):
        return None
    if level == "edge":
        return (
            "the server answers per molecule (one row, or one row per atom), and this "
            f"edge-level model ({task_type}) gives one row per bond, which it does not map to "
            "its molecule"
        )
    return (
        f"task type {task_type!r} names no output level (graph_, node_, edge_ or link_), so the "
        "server cannot tell what one row of its outputs is"
    )


class PackagedPredictor:
    """A deployment package's model, ready to predict from SMILES and structures."""

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
        # The app serves the input kinds the model accepts (models/input_contract.py, read from
        # the checkpoint, so older packages are checked too); with none it is not started
        packaged = cls(predictor, manifest)
        reason = packaged.unservable_reason()
        if reason is not None:
            raise DeploymentError(f"Cannot serve {package_dir}: {reason}")
        packaged.check_signature()
        # Requests compared with the training inputs (deployment/drift.py); after the signature
        # check, so only served molecules enter the window
        threshold, window = drift_settings()
        packaged.drift = DriftMonitor(manifest.get("input_reference"), threshold, window)
        return packaged

    @property
    def serves_identifiers(self) -> bool:
        """SMILES instances are served: the model accepts identifier inputs."""
        return self.predictor.input_contract.accepts(IDENTIFIER)

    @property
    def structure_example(self) -> list[dict[str, Any]]:
        """The package's verified structure example (``[]``: none recorded)."""
        return list(self.manifest.get("structure_example") or [])

    @property
    def serves_structures(self) -> bool:
        """Structure instances are served: the model accepts them and the package records a
        structure example, verified when the package was written (a structure path that was
        never verified is not served)."""
        return self.predictor.input_contract.accepts(STRUCTURE) and bool(self.structure_example)

    def unservable_reason(self) -> str | None:
        """Why the model cannot be served, or ``None`` when it can: its output level
        (:func:`output_level_refusal`), or no instance kind it accepts."""
        refusal = output_level_refusal(self.manifest.get("task_type"))
        if refusal is not None:
            return refusal
        if self.serves_identifiers or self.serves_structures:
            return None
        contract = self.predictor.input_contract
        reasons = [str(contract.refused.get(IDENTIFIER))]
        if contract.accepts(STRUCTURE):
            reasons.append(
                "the package records no structure example (its checkpoint was trained before "
                "MILIA recorded one), so structure inputs were not verified; retrain"
            )
        else:
            reasons.append(str(contract.refused.get(STRUCTURE)))
        return (
            f"the model accepts neither SMILES (identifier: {reasons[0]}) nor structure "
            f"instances (structure: {reasons[1]})"
        )

    @property
    def output_level(self) -> str | None:
        """What one row of the model's outputs describes, from the package's task type."""
        return output_level(self.manifest.get("task_type"))

    @property
    def output_names(self) -> list[str] | None:
        """The output column names of the package's signature (``None``: not recorded)."""
        names = ((self.manifest.get("signature") or {}).get("outputs") or {}).get("names")
        return list(names) if names else None

    def check_signature(self) -> None:
        """Featurize the package's examples as requests are featurized; compare with the signature.

        The model was trained on ``signature.inputs.node_features`` columns; MILIA code that
        featurizes differently would compute wrong values rather than fail, so the server refuses
        to start instead. Checked: the input example (a SMILES) when SMILES are served, the first
        structure example when structures are. A version-1 package has no signature and is not
        checked.

        Raises:
            DeploymentError: The featurized width or the output count differs from the signature.
        """
        signature = self.manifest.get("signature")
        if not signature:
            return
        examples: list[Any] = []
        if self.serves_identifiers:
            examples.append((self.manifest.get("input_example") or ["C"])[0])
        if self.serves_structures:
            examples.append(self.structure_example[0])
        for example in examples:
            self._check_example(signature, example)

    def _check_example(self, signature: dict[str, Any], example: Any) -> None:
        expected_inputs = (signature.get("inputs") or {}).get("node_features")
        expected_outputs = (signature.get("outputs") or {}).get("count")
        shown = example if isinstance(example, str) else example.get("name") or "a structure"
        (graph,) = self.featurize([example], observe=False)
        width = graph.x.size(-1) if getattr(graph, "x", None) is not None else None
        if expected_inputs is not None and width != expected_inputs:
            raise DeploymentError(
                f"This MILIA featurizes {shown!r} into {width} node features; the package's "
                f"model was trained on {expected_inputs}. Serve it with the MILIA version that "
                f"wrote it ({self.manifest.get('milia_version')})."
            )
        if expected_outputs is not None:
            per = "atom" if self.output_level == "node" else "molecule"
            rows = graph.num_nodes if per == "atom" else 1
            out = self.predictor.predict_batch([graph], return_numpy=True).reshape(rows, -1)
            if out.shape[1] != expected_outputs:
                raise DeploymentError(
                    f"The model returns {out.shape[1]} values per {per}; the package's "
                    f"signature records {expected_outputs}"
                )

    def featurize(self, instances: Sequence[Any], observe: bool = True) -> list[Any]:
        """Graphs for request instances, in order, prepared as the training graphs were.

        A string is a SMILES: featurized with the training run's ``structural_features_config``
        and passed through the training filters and pre-transforms (``Predictor.prepare_inputs``).
        A structure (a :class:`StructureInstance`, or its fields as a mapping) is built as the
        training dataset built its molecules (``Predictor.prepare_structures``). Observed by the
        drift monitor when ``observe``.

        Raises:
            ValueError: A SMILES cannot be parsed, or an instance's kind is not served; the
                message names its index.
            InputRefusedError: Instances refused (outside the training filters, a structure that
                cannot be built), every one named with its reason.
        """
        from milia_pipeline.models.post_training import convert_to_pyg
        from milia_pipeline.molecules.structure_inputs import StructureInput

        smiles_at, structures_at = [], []
        for index, instance in enumerate(instances):
            (smiles_at if isinstance(instance, str) else structures_at).append(index)
        refused = []
        if smiles_at and not self.serves_identifiers:
            reason = self.predictor.input_contract.refused.get(IDENTIFIER)
            refused += [(i, f"this model does not accept SMILES: {reason}") for i in smiles_at]
        if structures_at and not self.serves_structures:
            reason = self.unserved_structures_reason()
            refused += [(i, f"structures are not served: {reason}") for i in structures_at]
        if refused:
            raise InputRefusedError(refused)

        graphs: dict[int, Any] = {}
        refusals: list[tuple[int, str]] = []
        if smiles_at:
            config = self.predictor.structural_features_config
            featurized = []
            for index in smiles_at:
                try:
                    featurized.append(
                        convert_to_pyg(
                            instances[index],
                            format="smiles",
                            structural_features_config=config,
                            **self.predictor.featurization_kwargs,
                        )
                    )
                except ValueError as exc:
                    raise ValueError(f"instances[{index}]: {exc}") from exc
            try:
                graphs.update(
                    zip(
                        smiles_at, self.predictor.prepare_inputs(featurized, smiles_at), strict=True
                    )
                )
            except InputRefusedError as error:
                refusals += error.refusals
        if structures_at:
            structures = []
            for index in structures_at:
                instance = instances[index]
                fields = instance.model_dump() if isinstance(instance, BaseModel) else instance
                try:
                    structures.append(StructureInput.from_dict(fields))
                except ValueError as exc:
                    raise ValueError(f"instances[{index}]: {exc}") from exc
            try:
                graphs.update(
                    zip(
                        structures_at,
                        self.predictor.prepare_structures(structures, structures_at),
                        strict=True,
                    )
                )
            except InputRefusedError as error:
                refusals += error.refusals
        if refusals:
            raise InputRefusedError(sorted(refusals))
        ordered = [graphs[index] for index in range(len(instances))]
        if observe and self.drift is not None:
            self.drift.observe(ordered)
        return ordered

    def unserved_structures_reason(self) -> str:
        """Why structure instances are not served (when :attr:`serves_structures` is false)."""
        contract = self.predictor.input_contract
        if not contract.accepts(STRUCTURE):
            return str(contract.refused.get(STRUCTURE))
        return (
            "the package records no verified structure example (its checkpoint was trained "
            "before MILIA recorded one); retrain and repackage"
        )

    def predict_instances(self, instances: Sequence[Any], batch_size: int = 32) -> list[Any]:
        """Model outputs per instance (SMILES or structure), in request order: one row each, or
        one row per atom for a node-level model (:meth:`per_instance`).

        Raises:
            ValueError, InputRefusedError: As :meth:`featurize`.
        """
        graphs = self.featurize(instances)
        predictions = self.predictor.predict_batch(graphs, batch_size=batch_size, return_numpy=True)
        return self.per_instance(predictions, graphs)

    def per_instance(self, rows: Any, graphs: Sequence[Any], bound: bool = False) -> list[Any]:
        """``rows`` (the model's output rows for ``graphs``) grouped by instance: one row per
        graph, or for a node-level model each graph's atom rows (``num_nodes`` of them, in node
        order). ``bound``: interval bounds, a non-finite one as ``None`` (``json_bounds``).

        Raises:
            ValueError: The row count is not the graphs' (molecules, or atoms).
        """
        rows = np.asarray(rows)
        as_list = json_bounds if bound else (lambda part: part.tolist())
        if self.output_level != "node":
            return as_list(rows.reshape(len(graphs), -1))
        counts = [int(graph.num_nodes) for graph in graphs]
        if len(rows) != sum(counts):
            raise ValueError(f"{len(rows)} output rows for {sum(counts)} atoms")
        rows = rows.reshape(sum(counts), -1)
        return [as_list(part) for part in np.split(rows, np.cumsum(counts)[:-1])]

    def atomic_numbers(self, graphs: Sequence[Any]) -> list[list[int]] | None:
        """Each instance's atomic numbers in node order, for a node-level model (else ``None``)."""
        if self.output_level != "node":
            return None
        return [graph.z.tolist() for graph in graphs]

    def predict_smiles(self, smiles: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """One row of model outputs per SMILES, in input order (:meth:`predict_instances`)."""
        return self.predict_instances(smiles, batch_size)

    def predict_smiles_with_intervals(
        self, smiles: Sequence[str], batch_size: int = 32
    ) -> tuple[list[list[float]], dict[str, Any] | None]:
        """As :meth:`predict_instances_with_intervals` (kept for SMILES-only callers)."""
        return self.predict_instances_with_intervals(smiles, batch_size)

    def predict_instances_with_intervals(
        self, instances: Sequence[Any], batch_size: int = 32
    ) -> tuple[list[Any], dict[str, Any] | None]:
        """Rows as :meth:`predict_instances`, plus conformal intervals (shaped alike) when the
        package has them."""
        return self.predict_response(instances, batch_size)[:2]

    def predict_response(
        self, instances: Sequence[Any], batch_size: int = 32
    ) -> tuple[list[Any], dict[str, Any] | None, list[list[int]] | None]:
        """Rows and intervals as :meth:`predict_instances_with_intervals`, and each instance's
        atomic numbers for a node-level model (:meth:`atomic_numbers`)."""
        graphs = self.featurize(instances)
        if not self.predictor.calibration:
            rows = self.predictor.predict_batch(graphs, batch_size=batch_size, return_numpy=True)
            return self.per_instance(rows, graphs), None, self.atomic_numbers(graphs)
        result = self.predictor.predict_with_intervals(graphs, batch_size=batch_size)
        intervals = {
            "alpha": result["alpha"],
            "method": result["method"],
            "lower": self.per_instance(result["lower"], graphs, bound=True),
            "upper": self.per_instance(result["upper"], graphs, bound=True),
        }
        unbounded = unbounded_outputs(result["lower"], result["upper"])
        if unbounded:
            names = self.output_names
            intervals["unbounded"] = [names[i] if names else i for i in unbounded]
        return self.per_instance(result["mean"], graphs), intervals, self.atomic_numbers(graphs)


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
        too_long = [
            i
            for i, s in enumerate(request.instances)
            if isinstance(s, str) and len(s) > MAX_SMILES_LENGTH
        ]
        if too_long:
            raise HTTPException(
                status_code=422,
                detail=f"instances {too_long} exceed {MAX_SMILES_LENGTH} characters",
            )
        try:
            # Featurization and the forward pass are CPU-bound: run them off the event loop.
            rows, intervals, atomic_numbers = await run_in_threadpool(
                model.predict_response, request.instances
            )
        except InputRefusedError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": exc.message,
                    "refused": [{"index": i, "reason": r} for i, r in exc.refusals],
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictResponse(
            predictions=rows,
            model_name=model.manifest["model_name"],
            task_type=model.manifest["task_type"],
            output_names=model.output_names,
            output_level=model.output_level,
            atomic_numbers=atomic_numbers,
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


def require_servable(package_dir: str | Path) -> None:
    """Refuse a package whose model the server cannot serve (its output level, or neither SMILES
    nor structures).

    Decided as the server decides at start (:meth:`PackagedPredictor.unservable_reason`, the
    contract read from the package's checkpoint), so packages written before the contract are
    checked the same way.

    Raises:
        DeploymentError: The package is invalid, or no instance kind is served (the reason
            names what to do instead).
    """
    from milia_pipeline.models.post_training.inference.predictor import Predictor

    checkpoint, manifest = verify_package(package_dir)
    predictor = Predictor.from_checkpoint(
        checkpoint_path=checkpoint, working_root_dir=Path(package_dir), device=torch.device("cpu")
    )
    reason = PackagedPredictor(predictor, manifest).unservable_reason()
    if reason is not None:
        raise DeploymentError(
            f"Cannot serve {package_dir}: {reason}. Deploy it with the 'local' target instead."
        )


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

    Raises:
        DeploymentError: The package is invalid, or its model does not accept the identifier
            inputs the server takes (an image whose server would refuse to start is not written).
    """
    from milia_pipeline import __version__

    require_servable(package_dir)
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
