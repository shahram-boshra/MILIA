# MILIA Quick Start Guide

**For anyone with authenticated access to the MILIA private repository** — collaborators, peer reviewers, interviewers, students joining the project, and any other invitee.

This guide takes you from "I just accepted the repository invitation" to "I have run MILIA on my own machine and verified it works" in **roughly 30 minutes** on a CPU-only laptop, without contacting the authors. Every step states the expected output so you can self-verify.

> **Scope note (Linux-only).** This document covers Linux only — specifically Debian/Ubuntu and derivatives such as Linux Mint. macOS and Windows-WSL coverage is deliberately deferred; if you are on those platforms, the [`README.md`](README.md) `Installation` section and the upstream Docker / GitHub-CLI documentation linked below are sufficient to adapt the steps yourself.

> **Relationship to `README.md`.** `README.md` is the project reference — *what* MILIA is and *how* its modules fit together. This document is the onboarding tutorial — *the shortest reproducible path from zero to a working install*. Where the two would overlap, this guide cross-references `README.md` rather than duplicating it. The single deliberate exception is the GHCR authentication block in §3, which is embedded verbatim because it is a hard gate: you cannot continue past §3 until authentication succeeds, and a context switch to another document at that point breaks the linear-execution contract of this guide.

## Table of contents

1. [Welcome — what your authenticated access grants you](#1-welcome--what-your-authenticated-access-grants-you)
2. [Prerequisites — install Docker and GitHub CLI](#2-prerequisites--install-docker-and-github-cli) *(~10 min)*
3. [Authenticate to GHCR](#3-authenticate-to-ghcr) *(~3 min)*
4. [Pull and run the MILIA Docker image](#4-pull-and-run-the-milia-docker-image) *(~5 min)*
5. [The 5-minute health check](#5-the-5-minute-health-check) *(~3 min)*
6. [The 15-minute "is this software real?" walkthrough](#6-the-15-minute-is-this-software-real-walkthrough) *(~15 min)*
7. [Where to look in the codebase](#7-where-to-look-in-the-codebase)
8. [What to do if something fails](#8-what-to-do-if-something-fails)
9. [For peer reviewers — what MILIA claims and where to verify](#9-for-peer-reviewers--what-milia-claims-and-where-to-verify)

---

## 1. Welcome — what your authenticated access grants you

You have accepted an invitation to a **private** GitHub repository. That single act has quietly enabled three things that this guide is about to use:

1. **Read access to the source tree** at [`github.com/shahram-boshra/MILIA`](https://github.com/shahram-boshra/MILIA) — including `main.py`, the `milia_pipeline/` package, the `configs/` directory, and the test suite. You can browse, clone, and fork.
2. **Pull access to the private container image** published at `ghcr.io/shahram-boshra/milia:latest`. The image is hosted on **GitHub Container Registry (GHCR)** and inherits its permissions from this repository, so the same invitation that granted source access also granted image-pull access — but the registry will not give you the image until you authenticate (§3).
3. **The right to file issues** at [`github.com/shahram-boshra/MILIA/issues`](https://github.com/shahram-boshra/MILIA/issues) — the channel of record for anything that fails.

What this guide does **not** assume you have:

- An existing Docker installation, an existing `gh` CLI installation, or an existing Personal Access Token. §2 covers all three from a clean machine.
- A GPU. The shipped configuration in `configs/models.yaml` is preconfigured for low-resource execution — small batch size, few epochs, small ensembles — so a CPU-only laptop is sufficient for the full §6 walkthrough.
- Familiarity with MILIA's internals. §7 maps the codebase once you have a running container and want to look around.

If you ever want the *reference* picture rather than the onboarding path, [`README.md`](README.md) covers what MILIA is, why it exists, and what each of the 11 core modules does. This guide stays narrowly focused on getting you to a running install.



## 2. Prerequisites — install Docker and GitHub CLI

You need exactly two tools on the host machine: **Docker Engine** (to pull and run the MILIA container) and **GitHub CLI** (`gh`, the simplest way to authenticate to GHCR in §3). Total wall-clock time on a clean machine: 5–10 minutes, network-dependent.

> **Distribution scope.** The commands below are taken from the upstream Docker and GitHub CLI installation pages and target **Debian and Ubuntu**. Per the upstream Docker documentation, installation on Ubuntu derivatives such as **Linux Mint** is not officially supported by Docker but generally works when you follow the Ubuntu instructions and substitute your distribution's matching Ubuntu codename (for example, Linux Mint 22.x maps to Ubuntu 24.04 `noble`). For any non-Debian distribution (Fedora, Arch, RHEL, …) follow the matching section on the upstream pages linked at the end of this section — the rest of this guide (§3 onward) is distribution-agnostic.

### 2.1 Install Docker Engine

If a previous, distribution-packaged Docker is present, remove it first to avoid conflicts with the official packages:

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
    sudo apt-get remove -y $pkg
done
```

(Apt may report that none of these packages were installed — that is fine and expected on a clean machine.)

Add Docker's official apt repository (signed by Docker's GPG key):

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
```

> If you are on Linux Mint and `${UBUNTU_CODENAME:-$VERSION_CODENAME}` resolves to your Mint codename rather than the upstream Ubuntu one, replace that fragment with the Ubuntu codename your Mint release is built on (e.g. `noble` for Mint 22.x). Verify with `lsb_release -a` and the [Linux Mint release notes](https://linuxmint.com/download_all.php).

Install the engine and CLI:

```bash
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Add your user to the `docker` group so subsequent commands in this guide work without `sudo` (this matches the upstream post-install step for Linux):

```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Pass criterion**: the following must succeed without errors and without `sudo`:

```bash
docker version
docker run --rm hello-world
```

The second command pulls and runs the official `hello-world` image and prints a confirmation message. If both succeed you are done with Docker — proceed to §2.2.

### 2.2 Install GitHub CLI (`gh`)

Add the GitHub CLI apt repository (signed by GitHub's GPG key):

```bash
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install -y wget)) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && out=$(mktemp) \
  && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && sudo mkdir -p -m 755 /etc/apt/sources.list.d \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && sudo apt update \
  && sudo apt install -y gh
```

**Pass criterion**:

```bash
gh --version
```

prints a version string of the form `gh version X.Y.Z (YYYY-MM-DD)`. If you see that line, both prerequisites are satisfied — proceed to §3.

### 2.3 Authoritative sources

If anything in §2.1 or §2.2 has drifted from the upstream documentation since this guide was written, the upstream pages are the source of truth:

- Docker Engine on Ubuntu: <https://docs.docker.com/engine/install/ubuntu/>
- Docker Engine — overview of supported distributions: <https://docs.docker.com/engine/install/>
- Docker — post-install steps for Linux (rootless / `docker` group): <https://docs.docker.com/engine/install/linux-postinstall/>
- GitHub CLI installation (all platforms): <https://github.com/cli/cli/blob/trunk/docs/install_linux.md>



## 3. Authenticate to GHCR

This is the gate. Until you complete it, `docker pull ghcr.io/shahram-boshra/milia:latest` in §4 will fail with `unauthorized` or `denied: denied`. After it, every subsequent step in this guide works.

You have two equivalent options. **Option A (GitHub CLI)** is recommended for most users — your existing `gh` login does the work and no Personal Access Token has to be managed by hand. **Option B (Personal Access Token)** is the right choice in CI/CD pipelines, on shared machines, or anywhere you cannot or do not want to run an interactive `gh auth login` flow.

> **Why this block is duplicated from `README.md`.** The instructions in the green callout below appear verbatim in [`README.md` § Installation → Method 1: Docker (Recommended)](README.md#method-1-docker-recommended). They are reproduced here, rather than cross-referenced, because they are a hard prerequisite gate for the rest of this guide: a reader who must context-switch to `README.md` mid-flow to authenticate breaks the "linearly executable from zero" contract this document promises. The walkthrough commands in §6 are not gates and are therefore cross-referenced rather than embedded.
>
> **Maintainer invariant.** If the GHCR authentication blockquote in **either** `README.md` (§ Installation → Method 1) or this section is updated, the corresponding block in the other file MUST be updated in the same Git commit. The two blocks are identified canonically by the marker string `**Note (Private Repository):**` which begins each — a future programmatic enforcement mechanism (custom pre-commit hook, or CI grep check) can detect drift by diffing the two blocks identified by that marker. Until that mechanism exists, this paragraph is the only safeguard against drift.

### The authentication block (verbatim from `README.md` § Installation → Method 1)

> **Note (Private Repository):** MILIA's GHCR image is private. Before pulling, authenticate
> to GHCR using either [GitHub CLI](https://cli.github.com/) or a
> [Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
> with `read:packages` scope:
>
> ```bash
> # Option A: GitHub CLI (recommended — no PAT needed)
> gh auth login
> echo $(gh auth token) | docker login ghcr.io -u USERNAME --password-stdin
>
> # Option B: Personal Access Token with read:packages scope
> echo YOUR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
> ```

### Notes on the placeholders

- **`USERNAME`** / **`YOUR_GITHUB_USERNAME`** — replace with your actual GitHub username (case-insensitive, no `@`, the same string that appears in `https://github.com/<username>`). Run `gh api user --jq .login` after `gh auth login` if you are not sure.
- **`YOUR_PAT`** — Option B only. Generate a fine-grained or classic Personal Access Token at <https://github.com/settings/tokens>; the only scope you need for this guide is **`read:packages`**. Do not paste the token onto the command line in plain shells where it would land in history; either use a here-doc, set it via an environment variable read from a file, or use Option A.

### Pass criterion

The final command of either option prints exactly:

```
Login Succeeded
```

If you see that line, GHCR will accept the `docker pull` in §4. If you see `unauthorized`, `denied: denied`, or `incorrect username or password`, jump to §8.1 — do not retry blindly. **Do not proceed to §4 until `Login Succeeded` has been printed.**



## 4. Pull and run the MILIA Docker image

After §3 prints `Login Succeeded`, the registry will hand you the image. The pull is network-bound (a few hundred MB) and typically completes in 1–3 minutes on a residential connection.

### 4.1 Pull

```bash
docker pull ghcr.io/shahram-boshra/milia:latest
```

Expected output ends with a line of the form:

```
Status: Downloaded newer image for ghcr.io/shahram-boshra/milia:latest
ghcr.io/shahram-boshra/milia:latest
```

If you see `unauthorized` or `denied: denied` here, your §3 authentication did not stick — the most common cause is that `docker login ghcr.io` was run as a different OS user or without the `--password-stdin` line completing successfully. Re-run §3 (Option A or Option B) and try the pull again.

### 4.2 (Optional) Verify image integrity by digest

If you want a defence against registry tampering or a stale local cache, pin to the manifest digest published by the project's release workflow:

```bash
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/shahram-boshra/milia:latest
```

Compare the printed digest against the one in the corresponding GitHub Actions workflow run on the [Actions tab](https://github.com/shahram-boshra/MILIA/actions). For audited reproducibility you can pull by digest directly:

```bash
docker pull ghcr.io/shahram-boshra/milia@sha256:<digest-from-actions>
```

### 4.3 Run the container interactively

```bash
docker run --rm -it ghcr.io/shahram-boshra/milia:latest
```

What each flag does:

- `--rm` — delete the container's filesystem layer when you exit. Safe for a one-shot evaluation; omit it if you want to `docker start` the same container later.
- `-it` — attach an interactive TTY. Without this, the shell exits immediately.

When the run succeeds, the prompt changes to:

```
(shah_env) root@<container-id>:/app/milia#
```

The `(shah_env)` prefix tells you that MILIA's conda environment is already active. The working directory `/app/milia` contains the source tree; `/app` is the image's `WORKDIR`. **Every command from §5 onward is run from inside this prompt**, not from your host shell.

### 4.4 Pass criterion

Inside the container prompt, run:

```bash
milia --help
```

and verify that the first line is `usage: milia [-h] ...` followed by MILIA's argument groups. If `milia --help` prints help text without an error, the image is healthy and you can proceed to §5. If `milia: command not found`, the image was pulled but the entrypoint shell did not activate the conda env — exit and re-run §4.3 (the `-it` flag is required).

### 4.5 Building locally instead (optional)

If you would rather build the image from source than pull it — for example, to inspect or modify the `Dockerfile` — see [`README.md` § Installation → Method 1 → "Or build locally from the Dockerfile"](README.md#method-1-docker-recommended). The resulting image is interchangeable with the pulled one for everything in §5–§6.



## 5. The 5-minute health check

You are now inside the container with `(shah_env) root@…:/app/milia#` as your prompt. Before exercising the full pipeline, run the smoke suite — a curated subset of the 127-test suite (see [`README.md` § Testing](README.md#testing)) tagged with `@pytest.mark.smoke`, designed to fail fast if anything in the image is broken.

```bash
pytest -m smoke --tb=short
```

**Expected duration**: approximately 1–2 minutes on a typical CPU-only laptop. (This estimate is derived from the published `smoke-test` GitHub Actions job, which completes in roughly 2m24s end-to-end including `docker pull`; the pytest portion alone is shorter.)

### What a pass looks like

The final line of the output is of the form:

```
========================== N passed in T.TTs ==========================
```

where `N` is the number of smoke-tagged tests collected (currently 20+, defined in `tests/test__init__models_builders.py` and other test modules) and `T.TT` is the wall-clock time. **The exit code is `0`.** Anything else is a fail.

### What a fail looks like

Any of:

- The line above ends in `failed`, `error`, or `passed, X failed`.
- pytest exits non-zero.
- The collection phase prints `ERROR collecting …` before any test runs (this usually means a missing import — i.e. the image was built incompletely).

If you see any of those, **do not proceed to §6**. Jump to §8.4.

### What this gate proves and does not prove

A green smoke suite establishes that:

- The Python environment inside the image is internally consistent (PyTorch, PyTorch Geometric, RDKit, and MILIA's own modules all import).
- The model/dataset registries populate correctly at import time.
- The CLI entrypoint and configuration loader are functional.

It does **not** exercise dataset download, full training, or hyperparameter optimization — those are §6's job. A green smoke pass is necessary, not sufficient, evidence that MILIA is healthy on your machine.



## 6. The 15-minute "is this software real?" walkthrough

The smoke suite proved the image works. This section proves **MILIA itself works** — that it can ingest a dataset, build a graph representation, train a small GNN on a CPU, and predict on a fresh molecule.

The 7-step walkthrough is documented authoritatively in [`README.md` § "Trying MILIA — Reproducible Walkthrough"](README.md#trying-milia--reproducible-walkthrough). It is cross-referenced rather than duplicated here because, unlike §3's authentication gate, the walkthrough commands are not prerequisites for any later step in *this* document — by the time you reach §6 you already have a working terminal inside the container, so a context switch to `README.md` is safe.

**Two prerequisites already handled for you inside the Docker image**, so you can skip directly to README's "Step-by-step" subsection:

1. The `working_root_dir` configuration value (the only path you would normally have to set) is **preset inside the container** to `/root/Chem_Data/Milia_PyG_Dataset` — leave it as-is.
2. The shipped `configs/models.yaml` is preconfigured for low-resource execution (small batch size, few epochs, small ensembles), so the full sequence runs to completion on a CPU-only laptop.

**Minimal acceptance path** (≤ 15 min on CPU): execute steps 1, 2, 3, 5, and 7 from the README walkthrough — that is, smoke check, dry-run, dataset processing, training, and prediction. Step 4 (`milia --stats-only`) is informational and step 6 (`milia --train --hpo`) is the optional HPO variant of step 5.

### Pass criterion

After completing step 7 of the README walkthrough, the file `./predictions.csv` exists in the current directory and contains predicted property values for the five sample molecules shipped at `test_data/molecules.csv` (ethanol, acetic acid, benzene, isopropanol, triethylamine). Confirm with:

```bash
ls -lh ./predictions.csv && head -n 6 ./predictions.csv
```

If the file exists, has non-zero size, and the first line is a CSV header followed by five data rows, **you have completed the §5e.1 acceptance criterion** — pulled the image, run the smoke test, and executed one full walkthrough command, within roughly 30 minutes total.



## 7. Where to look in the codebase

You now have a running install. If you want to *understand* MILIA — to evaluate, contribute, or extend it — three files are the right starting point, in this order:

| Read first | Why |
|---|---|
| [`main.py`](https://github.com/shahram-boshra/MILIA/blob/main/main.py) | The single entrypoint. Tells you, in fewer than 100 lines, what the CLI dispatches to and the lifecycle of a single MILIA run. |
| [`milia_pipeline/cli_manager.py`](https://github.com/shahram-boshra/MILIA/blob/main/milia_pipeline/cli_manager.py) | ~3,800 lines covering 12 argument groups and 12+ processing modes — the authoritative reference for every flag you saw in §6. If you want to know *what* MILIA can do at the command line, this file is the contract. |
| [`configs/main.yaml`](https://github.com/shahram-boshra/MILIA/blob/main/configs/main.yaml) | The top-level YAML. All other files in `configs/` (descriptors, models, plugins, transformations, structural features, filter, and per-dataset configs under `configs/datasets/`) are deep-merged into this one at load time. If you want to know *how* MILIA is configured, this is the entry. |

Once those three are familiar, the comprehensive structural map is [`MILIA_Pipeline_Project_Structure.md`](MILIA_Pipeline_Project_Structure.md) — 4,400+ lines covering every directory, file, and class in the package, organised top-down. Use it as a lookup, not a linear read.

For a higher-level, narrative tour of the 11 core modules and the split-configuration architecture, see [`README.md` § Architecture](README.md#architecture) and [`README.md` § Datasets](README.md#datasets) — the latter covers the 10 shipped dataset implementations and the three-file pattern for adding your own.



## 8. What to do if something fails

This section catalogues the concrete failures you may hit at each gate of the guide and the remedy for each. If a failure is not listed here, file an issue per §8.7 — do not improvise around it.

### 8.1 §3 authentication fails (`docker login ghcr.io` does not print `Login Succeeded`)

**Most common symptoms**:

- `Error response from daemon: Get "https://ghcr.io/v2/": unauthorized`
- `incorrect username or password`
- `denied: denied` on the subsequent pull in §4

**Likely causes and remedies**:

1. **The `gh` CLI is not actually logged in.** Run `gh auth status` — if it does not say `Logged in to github.com as <your-username>`, run `gh auth login` first, then retry the `docker login` line from §3.
2. **Stale Docker credentials from a previous session.** Run `docker logout ghcr.io`, then re-run §3 from the top.
3. **PAT (Option B) missing the `read:packages` scope.** Re-issue the token at <https://github.com/settings/tokens> with `read:packages` checked, and retry. Tokens cannot be retroactively granted scopes.
4. **`USERNAME` placeholder not substituted.** The literal string `USERNAME` from the README block is a placeholder — replace it with your GitHub username before running.

### 8.2 §4 pull fails (`docker pull` returns an error)

**`unauthorized` / `denied: denied`** — §3 did not stick. Return to §8.1.

**`manifest unknown`** — the tag `latest` does not resolve. Confirm the URL is exactly `ghcr.io/shahram-boshra/milia:latest` (note: lowercase, no typos in the owner). If correct, the image may have been temporarily delisted by the maintainer; file an issue per §8.7.

**`net/http: TLS handshake timeout` or `dial tcp: i/o timeout`** — network or DNS problem on the host. Retry; if it persists, check `docker info` for proxy settings and corporate firewall rules. GHCR is served from `ghcr.io` and `pkg-containers.githubusercontent.com` — both must be reachable on TCP/443.

**`no space left on device`** — the Docker image is several hundred MB. Run `df -h /var/lib/docker` (or `docker system df`) and free space with `docker system prune` if needed.

### 8.3 §4.4 `milia --help` fails inside the container

**`milia: command not found`** — the conda environment did not activate. Symptom is usually a prompt without the `(shah_env)` prefix. Exit (`exit` or Ctrl-D) and re-run §4.3, ensuring the `-it` flags are present.

**`ImportError` or `ModuleNotFoundError` on a top-level package (torch, torch_geometric, rdkit, …)** — the image you pulled is corrupted or partial. Run `docker rmi ghcr.io/shahram-boshra/milia:latest` and re-pull from §4.1. If it recurs, file an issue per §8.7 and include the manifest digest from `docker inspect` (§4.2).

### 8.4 §5 smoke suite fails

**One or two tests `failed`** — likely environmental rather than image-broken. Re-run with verbose output: `pytest -m smoke -v --tb=long`. Capture the full output for §8.7.

**`ERROR collecting …` before any tests run** — a missing import. Treat the same as §8.3's `ImportError` case: re-pull the image first.

**Many tests `failed` simultaneously** — do not investigate individually. Capture the full pytest output and file an issue per §8.7; this is an image-level regression and the maintainer needs to see the whole picture.

### 8.5 §6 walkthrough fails

The README walkthrough (§ "Trying MILIA — Reproducible Walkthrough") is the authoritative source for command-by-command behaviour. If a specific README step fails:

- **Step 2 (`milia --dry-run`) fails** — almost always a `configs/` issue. Re-confirm the image's preset `working_root_dir` (§6 prerequisite 1) is intact: `grep -n working_root_dir configs/main.yaml` should print one line. If you have edited `configs/main.yaml` inside the container, revert by exiting and starting a fresh container.
- **Step 3 (`milia --process`) fails on dataset download** — connectivity from inside the container to the dataset host. Same diagnosis as §8.2's TLS/DNS branch.
- **Step 5 (`milia --train`) hangs or runs out of memory** — the shipped low-resource preset should not OOM on a typical laptop, but if it does, check `docker stats` from your host shell and confirm the Docker daemon's memory limit. Reduce nothing in `configs/models.yaml` without first reading `configs/models.yaml`'s comments — the preset values are interrelated.
- **Step 7 (`milia --predict`) writes no `predictions.csv`** — the most common cause is `--model-path ./checkpoints/best.pt` not resolving because step 5 did not actually train (check the console output from step 5 for an end-of-training summary).

For anything else in §6, file an issue per §8.7.

### 8.6 Resetting to a clean state

If you want to start over from scratch:

```bash
# from your host shell, NOT inside the container
docker logout ghcr.io
docker rmi ghcr.io/shahram-boshra/milia:latest
docker system prune -f
```

Then resume from §3.

### 8.7 Filing a useful issue

When opening an issue at <https://github.com/shahram-boshra/MILIA/issues>, include all of the following — anything less makes the maintainer's job harder and slows down your reply:

1. **Which step failed** — by section number from this guide (e.g. "§5 smoke suite, three tests failed").
2. **The image digest** — output of `docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/shahram-boshra/milia:latest` (from §4.2). Without this, the maintainer cannot reproduce.
3. **Host OS** — output of `lsb_release -a` (Linux) and `docker version --format '{{.Server.Version}}'`.
4. **The complete error output** — copy-paste, not paraphrased; redact any tokens, usernames, or local paths if they would be sensitive.
5. **What you tried** — which §8 subsection's remedies you ran and the result.



## 9. For peer reviewers — what MILIA claims and where to verify

If you are evaluating MILIA for publication review, hiring, or collaborative onboarding, the questions you most likely need to answer are:

1. **Does the software actually do what the README says it does?**
2. **Where in the source can I confirm each claim with my own eyes?**

This section maps the headline claims from [`README.md` § Key Features](README.md#key-features) and § Architecture to concrete locations in the codebase. No claim below is original to this guide — they are reproduced from `README.md` and pointed at their verification site.

| Claim (paraphrased from README) | Verify by inspecting |
|---|---|
| **No-code, YAML-driven ML/DL pipeline.** Run dataset curation, transformations, training, HPO, and prediction through configuration alone. | The 7 step-by-step commands you executed in §6, plus `configs/main.yaml` and the per-dataset files under `configs/datasets/` (10 files; see `MILIA_Pipeline_Project_Structure.md`). No Python is written by the user in §6. |
| **10 shipped dataset implementations** (VQM24 family, QM9, ANI-1x/1ccx/2x, rMD17, xxMD, QDπ). | `milia_pipeline/datasets/implementations/` — one `.py` per dataset. README § Datasets table names each, with primary-source citation. |
| **Every PyTorch Geometric model is reachable by name** via dynamic introspection. | `milia_pipeline/models/` — registry and factory. README § Key Features → "Unlimited Model Flexibility" + § Architecture row "models" (~25 files). |
| **400+ molecular descriptors across 6 categories.** | `milia_pipeline/descriptors/` — registry, 6+ files. README § Key Features → "Molecular Descriptors" enumerates the 6 categories with counts. |
| **Hardware-agnostic** — CPU, CUDA, MPS, TPU; 4 distributed strategies. | `milia_pipeline/models/` acceleration submodule. README § Key Features → "Hardware Agnostic" + Architecture row "models". The Docker image you ran is the CPU configuration. |
| **Optuna-based HPO** with 5 search algorithms and 5 pruning strategies; NAS for GNNs. | `milia_pipeline/models/hpo/` — 12 files. README § Key Features → "Advanced Hyperparameter Optimization" + Architecture row "models/hpo". |
| **Three-tier plugin architecture** (descriptors, transformations, models) with YAML manifests. | `plugins/` directory at the repo root. README § Key Features → "Three-Tier Plugin Architecture" + § Datasets → "Adding a dataset" subsection describes the three-file extension pattern. |
| **Pydantic V2 schema-validated configuration** with deep merge across split YAML files. | `milia_pipeline/config/` — 7 files. README § Key Features → "Flexible Configuration System" + Architecture row "config". |
| **127 unit and integration tests** covering all core modules. | `tests/` directory. README § Testing. The smoke subset you ran in §5 is the fast-failing prefix of this suite. |
| **30+ pre-registered PyG transforms; 7-layer transformation system; 3 validation levels × 5 scopes.** | `milia_pipeline/transformations/` — 4 files, ~16K lines. README § Key Features → "Extensible Graph Transformation System" + Architecture row "transformations". |
| **12 CLI argument groups; 12+ processing modes; interactive mode.** | `milia_pipeline/cli_manager.py` — single file, ~3,800 lines. README § Architecture (row "cli_manager"). |
| **Per-dataset capability matrix is registry-driven, not hard-coded.** Eight feature flags (vibrational, uncertainty, atomization, orbital, HOMO–LUMO, plus three more) gate behaviour without consumers learning dataset names. | README § Datasets → "Per-dataset capability matrix". Each `BaseDataset` subclass declares an immutable `DatasetFeatures` record; queried at runtime via `_get_dataset_feature(dataset_type, feature_name)`. |

### Independent reproducibility checklist

These are the things a peer reviewer can re-verify entirely from a freshly-pulled image, without contacting the authors:

- **The image you pulled is the one CI built.** Compare `docker inspect --format='{{index .RepoDigests 0}}'` (§4.2) against the digest published in the GitHub Actions run that produced `latest`.
- **The shipped configuration is laptop-runnable.** Step 5 (`milia --train`) of the README walkthrough completes on a CPU within the §5e.1 30-minute budget — your own clock is the witness.
- **A trained model produces predictions on unseen molecules.** `predictions.csv` from §6 contains 5 rows for 5 SMILES strings the model was not trained on (sample at `test_data/molecules.csv`).
- **The test suite is real.** `pytest` (without the `-m smoke` filter) inside the container exercises the full 127-test suite. README § Testing documents the markers; you do not have to take the README's word for the count — `pytest --collect-only -q | tail -n 1` prints it.

### Out of scope for this guide

- **Scientific accuracy of trained models** on benchmark datasets — that is a research question, not an onboarding question. The shipped low-resource preset is sized for *checking that the pipeline runs*, not for SOTA reproduction. For reproduction-grade configurations, contact the authors.
- **Performance benchmarking** (throughput, memory, scaling). The Docker image is single-CPU configured.
- **License compatibility review** — see [`LICENSE`](LICENSE) at the repo root.
- **Citation in published work** — citation metadata is in [`CITATION.cff`](CITATION.cff); see also [`README.md` § Citation](README.md#citation).

---

**End of Quick Start.** If you reached this point with `predictions.csv` written and the smoke suite green, you have completed the §5e.1 acceptance criterion — pulled the image, run the smoke test, and executed one full walkthrough command, within 30 minutes on a CPU-only laptop, without contacting the authors.

For everything beyond that, the canonical references are [`README.md`](README.md), [`MILIA_Pipeline_Project_Structure.md`](MILIA_Pipeline_Project_Structure.md), and [`CONTRIBUTING.md`](CONTRIBUTING.md).
