"""Build-time verification for MILIA container images (run by the Dockerfile).

Executed by every stage that installs the project (``builder`` and ``test``) so the checks are defined
once. A non-zero exit fails the image build early.

Checks:
  (a) the ``milia-py`` distribution is installed (metadata resolves) and ``milia_pipeline`` imports.
      uv installs the project editable by default (PEP 660), so the package legitimately resolves to
      the baked ``/app`` source tree;
  (b) the compiled PyG companion kernels load and run against the installed torch build (ABI check).
"""

import importlib.metadata

import torch
import torch_scatter

import milia_pipeline


def main() -> None:
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


if __name__ == "__main__":
    main()
