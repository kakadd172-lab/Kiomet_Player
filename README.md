# Kiomet Player

Kiomet Player is an experimental, local-first research project exploring how an AI system can observe and understand the real-time strategy game [Kiomet](https://kiomet.com/).

The project is being built incrementally. Its current work focuses on collecting visual observations, identifying game entities, organizing reference assets and metadata, and testing local vision-model-based game-state extraction. It is **not yet a complete or autonomous game-playing agent**.

> **Unofficial project:** Kiomet Player is an independent community project. It is not affiliated with, endorsed by, or sponsored by SoftbearStudios. Kiomet and its associated names and artwork belong to their respective owners.

## Project goals

The long-term goal is to develop a modular AI player with a clear separation between:

1. **Observation** — capture the visible game screen.
2. **Perception** — classify the screen and identify visible entities and interface information.
3. **Knowledge** — maintain structured references for towers, units, icons, and other game concepts.
4. **State extraction** — turn visual observations into structured, uncertainty-aware game-state data.
5. **Decision-making** — reason about the current game state and choose actions.
6. **Interaction** — carry out permitted actions through the game interface.

The first four areas are the focus of the current prototype. Strategy and automated interaction are future work, not capabilities this repository should be assumed to have today.

## Current status

This repository is an early-stage prototype. The tracked code currently includes:

- Browser-based Kiomet screen capture and observation scripts.
- Experiments that send screenshots to a locally running vision model through Ollama.
- Screen-classification and vision/perception experiments.
- Tower and unit reference metadata.
- Official Kiomet SVG icon assets, a source manifest, and scripts for rendering and preparing those assets.
- Initial recognition package structure and registry code.

Some recognition modules are currently placeholders or incomplete. A file or package existing in the repository does not mean its feature is implemented or reliable. Results from vision models should be treated as estimates and checked against the screenshot.

## Repository layout

| Path | Purpose |
| --- | --- |
| `browser_test.py` | Opens Kiomet in Chromium and captures an initial screenshot. |
| `perception/` | Screenshot observation, screen classification, vision experiments, and game-state extraction. |
| `kiomet_knowledge/assets/towers/official_svg/` | Kiomet tower SVG reference assets. |
| `kiomet_knowledge/assets/troops/official_svg/` | Kiomet unit SVG reference assets. |
| `kiomet_knowledge/assets/derived/` | Rendered PNG references and contact sheets derived from the SVG assets. |
| `kiomet_knowledge/recognition/` | Icon manifest, metadata, registries, and recognition package structure. |
| `kiomet_knowledge/tools/` | Asset rendering and metadata-generation utilities. |
| `extract_kiomet_icons.py` | Interactive browser-based icon extraction utility. |
| `official_source_test/` | Research notes and source excerpts used while investigating the upstream project. |

## Requirements

- Windows, Linux, or macOS.
- Python 3 and pip.
- Chromium managed by Playwright.
- [Ollama](https://ollama.com/) only if you want to run the local vision-model game-state extraction experiment.

The Python scripts use these packages:

- `playwright` — browser automation and screenshot capture.
- `Pillow` — image processing and contact-sheet generation.
- `requests` — requests to the local Ollama API.
- `numpy` and `opencv-python` — deterministic image matching and colour analysis.

## Setup

Run these commands from the repository root.

### 1. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
```

### 3. Optional: prepare the local vision model

Install and start Ollama, then pull the model expected by the current game-state extraction script:

```bash
ollama pull qwen2.5vl:3b
```

The script currently expects Ollama's local API at `http://localhost:11434` and the model name `qwen2.5vl:3b`. If you change either setting, update `perception/game_state_extractor.py` accordingly.

## Running the experiments

Run commands from the repository root, with the virtual environment activated.

### Capture one initial screenshot

```bash
python browser_test.py
```

This opens Kiomet in Chromium and saves an initial screenshot under `screenshots/`.

### Run the continuous observer

```bash
python perception/observer.py
```

The observer captures the visible browser screen repeatedly. Stop it with **Ctrl+C** in the terminal.

### Extract a game-state estimate

Start Ollama first, open Kiomet and capture a screenshot, then run:

```bash
python perception/game_state_extractor.py
```

The script selects the newest `screen_*.png` file in `screenshots/` and asks the configured local vision model to return structured JSON. The output is experimental: visual models can miss, misclassify, or misread game elements.

### Run the labelled icon-matcher benchmark

The benchmark compares manually labelled screenshot crops against all registered tower or unit templates. It writes a JSON report to the ignored `benchmark_results/` directory:

```bash
python perception/icon_matcher_benchmark.py
```

Useful options include `--top-k 5`, `--min-score 0.65`, `--min-margin 0.10`, `--labels path/to/labels.json`, and `--output path/to/report.json`. The score thresholds are provisional heuristics, not probabilities. The current dataset is deliberately small; do not treat its accuracy as a reliable estimate of in-game performance. The clipped King crop remains disabled until it is recaptured.

Run the automated tests with:

```bash
python -m unittest discover -s tests -v
```

### Work with the reference icons

The icon extraction utility is interactive and expects you to navigate to the relevant in-game galleries:

```bash
python extract_kiomet_icons.py
```

To render the checked-in SVG assets and rebuild their contact sheets:

```bash
python kiomet_knowledge/tools/render_transparent_icons.py
```

To regenerate the tower and unit metadata after the required SVG and PNG files are present:

```bash
python kiomet_knowledge/tools/build_metadata.py
```

## Screenshots and generated files

Screenshots and temporary diagnostics are intentionally excluded from this repository by `.gitignore`. The project's screenshot workflow is maintained separately from the main source repository. Do not assume that a screenshot created locally has been committed or uploaded.

Generated icon renders are kept under `kiomet_knowledge/assets/derived/` because they are reference data used by the current metadata workflow.

## Development principles

- **Local-first experimentation:** vision-model experiments are designed to run against a local Ollama service.
- **Evidence over guesses:** extracted state should distinguish visible evidence from uncertain interpretation.
- **Modular design:** observation, perception, knowledge, decision-making, and interaction should remain separable.
- **Traceable data:** reference assets should retain their source and licence information.
- **Honest capability claims:** incomplete modules and experimental model output must not be presented as finished functionality.

## Third-party material and attribution

The tower and unit SVG reference assets in this repository were extracted from Kiomet's rendered galleries. The upstream project is maintained by [SoftbearStudios](https://github.com/SoftbearStudios/kiomet), whose repository identifies its licence as **GNU Affero General Public License v3.0 (AGPL-3.0)**.

- Upstream source: [SoftbearStudios/kiomet](https://github.com/SoftbearStudios/kiomet)
- Game: [kiomet.com](https://kiomet.com/)
- Asset inventory and per-asset provenance: `kiomet_knowledge/recognition/official_icon_manifest.json`

The PNG reference images and contact sheets in `kiomet_knowledge/assets/derived/` are generated from those SVG assets and retain the same upstream provenance. Kiomet's original assets, branding, and trademarks are not claimed as original work by this project.

The upstream licence does not mean SoftbearStudios endorses this project. See [LICENSE](LICENSE) for the licence applicable to this repository and retain the provenance information when redistributing the third-party assets.

## Licence

Unless a file or asset carries a different notice, the original source code and project-authored material in this repository are made available under the **GNU Affero General Public License, version 3 (AGPL-3.0)**. The full licence text is in [LICENSE](LICENSE).

Third-party material remains subject to its own applicable notices and licence terms. In particular, the Kiomet-derived SVG and PNG reference assets are attributed to SoftbearStudios as described above. The repository licence does not transfer ownership of third-party material or grant rights to use third-party trademarks.

---

This project is intended for research, learning, and experimentation. Users are responsible for ensuring that their use of Kiomet and any automation complies with the game's rules and applicable terms.
