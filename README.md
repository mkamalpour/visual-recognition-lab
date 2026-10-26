# Visual Recognition Lab

[Explore the interactive project presentation](docs/overview.html).

Explore how CLIP matches an image to candidate descriptions, and how a supervised CNN learns CIFAR-10 classes. The local app includes lightweight recognition-math tools; model downloads and training belong in the optional Colab workflow. This is a university project, not a reproduction of the paper benchmark.

## Try it first

Start with the equation walkthrough in `notebooks/recognition_walkthrough.ipynb`, which uses small synthetic inputs. For real image recognition, use already-cached CLIP weights or the Colab guide below. Candidate labels come from you; the app does not discover an unrestricted vocabulary.

<details>
<summary>Using the existing managed Windows workspace</summary>

Assigned environment: **base**. From this repository folder in PowerShell:

```powershell
.\run.cmd --check
.\run.cmd
```

The repo-local launcher selects `G:\AnaConda\install`
and imports required startup modules. It installs nothing and downloads
no models. On a standalone clone where that interpreter is absent, it uses your
active `python`; activate a compatible environment first. `--check` reports the
actual interpreter; it does not validate optional models or every workflow.

</details>

## Run locally

Use an existing Conda environment. No installation, download, or new environment is performed automatically.

```powershell
conda activate base
python -m streamlit run app.py --server.address 127.0.0.1
```

After activation, `run.bat` runs the same command. The `ml` environment supports the lightweight interface and equation workflows too. Missing model packages or weights require Colab; they are never replaced with fabricated predictions.

On Windows, `run.bat --check` checks the active environment without opening the app. Run `run.bat` to start, or `run.bat --server.port 8502` if another app is using the default port. The launcher works from any working directory and installs nothing.

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

Original personal documents, images, datasets, and model weights are excluded.

The CNN tab labels the last completed training report and records its settings. Changing controls marks the report as belonging to the previous run; a failed rerun preserves that report. Model-free UI checks use a simulated training callback, not real CNN execution. Latest Windows / existing Conda base checks: 14 passed, 2 optional backend tests skipped.

Embedding validation rejects complex-valued inputs before conversion and checks classifier index shape, integer type, uniqueness and label bounds before ranking.

See the [engineering audit](docs/ENGINEERING_AUDIT.md) for the dated test baseline,
project assessment, environment assignment and remaining work.

## Additional workflow improvements

Per-class precision/recall/F1, balanced accuracy, macro F1 and confused-label diagnostics with downloadable JSON and stale-result protection.

Evaluation & method → Evaluate labelled predictions → Download evaluation report.
