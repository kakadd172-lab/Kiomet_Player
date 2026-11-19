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
- Category-aware tower and unit icon recognizers with conservative confidence handling.
- A screenshot-region recognition layer that crops caller-proposed regions and returns identities, candidate rankings, and screen coordinates.

The screenshot-region recognizer does **not** detect objects by itself: it requires bounding-box proposals from a separate detector or vision system. The current labelled crop dataset is small, so matching thresholds and scores remain experimental rather than calibrated probabilities. Results must be validated against real gameplay screenshots.

## Repository layout

| Path | Purpose |
| --- | --- |
| `browser_test.py` | Opens Kiomet in Chromium and captures an initial screenshot. |
| `perception/` | Screenshot observation, screen classification, region-based entity recognition, vision experiments, and game-state extraction. |
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

### Classify proposed regions in a screenshot

The region recognizer can be used by a future detector to classify boxes without coupling object detection to icon matching. Coordinates are normalized to the screenshot dimensions:

```python
from perception.entity_recognizer import RegionProposal, ScreenEntityRecognizer

recognizer = ScreenEntityRecognizer()
observations = recognizer.recognize_regions(
    "screenshots/screen_example.png",
    [RegionProposal("candidate-1", "tower", 0.25, 0.30, 0.06, 0.08)],
    method="shape",
)
for observation in observations:
    print(observation.display_name, observation.status, observation.bbox)
```

A proposal category must be `tower` or `unit`. Each result includes normalized and pixel bounding boxes, ranked candidates, similarity scores, and optional dominant colour. A rejected/ambiguous result has no accepted identity. This API classifies supplied regions; it does not find regions in a full screenshot.

### Run experimental full-screen entity detection

The entity detector asks the configured local vision model to propose tower/unit bounding boxes. By default it makes one resized full-screen request; pass `--tiled` to use up to four overlapping regions, which preserves more detail per request at the cost of additional inference calls. Each proposal is checked by the deterministic icon recognizers. The JSON report includes client wall-clock time, Ollama's reported total/load/prompt-evaluation/generation durations, token counts, source dimensions, and recognition-pipeline timings when the installed Ollama version returns those metrics. An annotated screenshot is also saved under the ignored `benchmark_results/entity_detection/` directory:

```bash
python -m perception.run_entity_detection
```

Pass `--image path/to/screenshot.png` to select a screenshot, `--model qwen3-vl:4b-instruct` to select the Ollama model, `--tiled` to test higher-detail regional proposals, `--max-image-side 1280` to change the full-frame resize limit, or `--method grayscale` to change the icon-matching method. The default model is `qwen3-vl:4b-instruct`; it must already be available in Ollama. Green boxes indicate accepted icon identities and amber boxes indicate uncertain matches. This is an experimental proposal pipeline: the vision model can miss objects or propose false boxes, and deterministic matching cannot correct a badly placed crop. Validate detections against manually labelled full screenshots before relying on them.

### Run the labelled icon-matcher benchmark

The benchmark compares manually labelled screenshot crops against all registered tower or unit templates. It writes a JSON report to the ignored `benchmark_results/` directory:

```bash
python perception/icon_matcher_benchmark.py
```

Useful options include `--top-k 5`, `--method grayscale` or experimental `--method shape`, `--min-score 0.65`, `--min-margin 0.10`, `--labels path/to/labels.json`, and `--output path/to/report.json`. The grayscale method is the original pixel-comparison baseline. Shape mode compares several foreground extractions and a dominant-connected-component silhouette. The latter can discard detached count markers beneath an icon before comparing normalized binary shapes. This reduces interference from coloured team markers, glow, and detached UI marks, but remains experimental and must be evaluated against a broader labelled screenshot set before being trusted. The score thresholds are provisional heuristics, not probabilities. The current dataset is deliberately small; do not treat its accuracy as a reliable estimate of in-game performance. The clipped King crop remains disabled until it is recaptured.

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
