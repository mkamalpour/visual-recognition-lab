# Visual Recognition Workbench

Educational CLIP recognition and supervised CIFAR-10 CNN training, adapted from an E-FineR university project. Candidate names are supplied by the user; automatic vocabulary discovery and paper benchmark accuracy are not claimed.

## Run locally

Use an existing Conda environment. No installation, download, or new environment is performed automatically.

```powershell
conda activate base
python -m streamlit run app.py --server.address 127.0.0.1
```

After activation, `run.bat` runs the same command. The `ml` environment supports the lightweight interface and equation workflows too. Missing model packages or weights require Colab; they are never replaced with fabricated predictions.

## Heavy work in Colab

Open [notebooks/colab_workbench.ipynb](notebooks/colab_workbench.ipynb), upload a ZIP of this repository, and run only the setup/model/training cells you need. Downloads are explicit. The Colab notebook has not been executed locally. The [equation walkthrough](notebooks/recognition_walkthrough.ipynb) contains executed synthetic examples, not image-accuracy measurements.

## Layout

```text
app.py                 Streamlit launcher
run.bat                Windows launcher using the active environment
vision_workbench/      Recognition math, CLIP, CNN, image preparation, interface
notebooks/             Colab workflow and executed equation walkthrough
requirements/          Optional UI and model dependency specifications
tests/                Focused regression tests
docs/                 Method, usage details, verification and limitations
```

## Workflows and checks

The four interface tabs cover candidate-based CLIP recognition, independent reference images and augmentation, supervised CNN checkpoints, and exact-label evaluation. Query/reference duplicates are rejected, stale results are marked, checkpoints are protected, and scores are labelled as relative scores.

Run lightweight checks from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m vision_workbench.cnn --help
```

Optional Torch tests skip when Torch/torchvision are unavailable. No model downloads occur in tests. See [method and verification](docs/method-and-verification.md) for measured checks, existing Conda capabilities, model setup, privacy and known limitations. Model caches respect HF_HOME/HF_HUB_CACHE; the maintained local setup points these to F:\DevCache.

Original personal documents, images, datasets, and model weights are excluded. No license has been selected.
