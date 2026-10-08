"""Build-time verification for MILIA container images (run by the Dockerfile).

Executed by every stage that installs the project (``builder`` and ``test``) so the checks are defined
once. A non-zero exit fails the image build early.

Checks:
  (a) the ``milia-py`` distribution is installed (metadata resolves) and ``milia_pipeline`` imports.
      uv installs the project editable by default (PEP 660), so the package legitimately resolves to
      the baked ``/app`` source tree;
  (b) the compiled PyG companion kernels load and run against the installed torch build (ABI check);
  (c) ``--accel <variant>``: the installed torch build matches the image variant (PA-0b) — ``cpu`` has
      no CUDA runtime, ``cuXYZ`` bundles CUDA ``X.Y`` (e.g. ``cu124`` → ``12.4``);
  (d) ``--compile`` (runtime stage only, where the C/C++ toolchain is installed): ``torch.compile`` with
      the default Inductor backend builds and runs a CPU kernel. Inductor generates C++ and raises
      ``InvalidCxxCompiler`` when no working compiler exists, so this proves the published image can
      run compiled models (PA-0b);
  (e) the ``hpo-postgres`` extra is installed: SQLAlchemy's ``postgresql+psycopg`` dialect loads its
      driver (psycopg 3 with its libpq), so the image can join an HPO study on PostgreSQL (P2-4b).
"""

import argparse
import importlib.metadata
import re

import torch
import torch_scatter

import milia_pipeline


def expected_cuda_version(accel: str) -> str | None:
    """Return the CUDA version a torch build for ``accel`` reports (``torch.version.cuda``).

    ``cpu`` → ``None``; ``cu<major><minor>`` → ``"<major>.<minor>"`` (the last digit is the minor
    version: ``cu118`` → ``11.8``, ``cu124`` → ``12.4``). Any other value is a configuration error.
    """
    if accel == "cpu":
        return None
    match = re.fullmatch(r"cu(\d+)(\d)", accel)
    if match is None:
        raise SystemExit(f"Unknown accelerator variant '{accel}': expected 'cpu' or 'cu<digits>'")
    return f"{int(match.group(1))}.{match.group(2)}"


def check_accelerator(accel: str) -> None:
    expected = expected_cuda_version(accel)
    actual = torch.version.cuda
    if actual != expected:
        raise SystemExit(
            f"torch build mismatch for ACCEL={accel}: torch.version.cuda={actual!r}, "
            f"expected {expected!r} (torch {torch.__version__})"
        )
    print(f"OK accelerator {accel}: torch.version.cuda={actual!r}")


def check_compile() -> None:
    def double(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    result = torch.compile(double)(torch.ones(2)).tolist()
    if result != [2.0, 2.0]:
        raise SystemExit(f"torch.compile check failed: got {result}, expected [2.0, 2.0]")
    print("OK torch.compile (Inductor, CPU)")


def check_postgres_driver() -> None:
    from sqlalchemy.engine import make_url

    dialect = make_url("postgresql+psycopg://").get_dialect()
    driver = dialect.import_dbapi()
    print(
        f"OK PostgreSQL driver: {dialect.name}+{dialect.driver} -> psycopg {driver.__version__} "
        f"({driver.pq.__impl__}, libpq {driver.pq.version()})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--accel", help="image variant to verify the torch build against (cpu, cu124, ...)"
    )
    parser.add_argument(
        "--compile", action="store_true", help="also verify torch.compile (needs a C/C++ compiler)"
    )
    args = parser.parse_args()

    print(
        "milia-py",
        importlib.metadata.version("milia-py"),
        "importable from",
        milia_pipeline.__file__,
    )
    src = torch.tensor([1.0, 1.0, 1.0, 1.0])
    index = torch.tensor([0, 0, 1, 1])
    result = torch_scatter.scatter_add(src, index, dim=0).tolist()
    if result != [2.0, 2.0]:
        raise SystemExit(
            f"torch_scatter.scatter_add ABI check failed: got {result}, expected [2.0, 2.0]"
        )
    print("OK torch", torch.__version__, "| torch_scatter", torch_scatter.__version__)
    check_postgres_driver()
    if args.accel is not None:
        check_accelerator(args.accel)
    if args.compile:
        check_compile()


if __name__ == "__main__":
    main()
