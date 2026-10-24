# Visual Recognition Lab

An educational visual-recognition application based on a university E-FineR study project. It separates pretrained CLIP image–text recognition from supervised CIFAR-10 CNN training. This repository is an adaptation, not a reproduction of the paper's benchmark results.

## Research context

The paper is *Vocabulary-free Fine-grained Visual Recognition via Enriched Contextually Grounded Vision-Language Model*, Demidov et al., ICCV Workshops 2025, pp. 4216–4225. See the [authors' official repository](https://github.com/demidovd98/e-finer) for their implementation and experiments.

The original project supplied fixed domain vocabularies and hand-authored descriptions. Such classification is zero-shot candidate matching; it does not establish automatic vocabulary discovery. A randomly initialized text transformer is not aligned with a separately supervised CNN. The revised system keeps those model families separate and labels results according to their actual method.

## Correctness changes

- Fit contextual classifiers from reference images independently of query images; query images must not become their own reference set.
- Normalize individual description embeddings before averaging them.
- Refine candidates using reference-image similarity; form separate visual prototypes from each candidate's pseudo-labelled reference images.
- Use text-only weights for candidates with no assigned reference images.
- Present temperature-scaled scores as relative candidate scores, not calibrated certainty or measured accuracy.
- Surface invalid inputs and model failures instead of returning zero embeddings or fabricated error predictions.
- Keep model downloads, training, checkpoints, datasets, and private images out of source control.

The original notebook, report, article, and media remain outside this repository. No license has been selected.

## Setup and run

Use your existing Conda environment; no new environment is required and no local installation is performed automatically.

```powershell
conda activate base
python -m streamlit run app.py --server.address 127.0.0.1
```

`ml` can run the lightweight interface and NumPy workflows too. After activation, `run.bat` is an optional launcher. Heavy training and pretrained models belong in **Colab**: upload this repository and open `notebooks/colab_workbench.ipynb`. Its setup and download cells are explicit and were not executed locally. Requirements files describe the optional Colab model stack, not a request to modify your existing environments.

Local environment inspection found:

| Environment | Python | Relevant installed packages | Missing for this project |
|---|---|---|---|
| base | 3.13.5 | NumPy 2.1.3, Pillow 11.1.0, Streamlit 1.45.1, Torch 2.14.0, Transformers 4.53.3 | torchvision |
| ml | 3.10.19 | NumPy 1.26.4, Pillow 11.3.0, Streamlit 1.57.0 | Torch, torchvision, Transformers |
| cnn | 3.10.20 | NumPy 1.26.4, Pillow 12.2.0, Gradio 6.15.2 | Torch, torchvision, Transformers, Streamlit (the existing TensorFlow-GPU 2.10 stack was verified separately) |

These environments were inspected without installing or changing anything in them. Package presence does not guarantee every optional backend works. Heavy inference and full CIFAR-10 training remain unverified; use Colab for those operations.

Open **CLIP recognition**, upload a query, and choose a domain or custom candidate names. To obtain missing pretrained weights, explicitly enable **Allow pretrained model download** before running. This respects HF_HUB_CACHE/HF_HOME; on the maintained local system they point to F:\DevCache\huggingface. Colab uses its own cache. Downloaded weights are not committed. Weights can be large. Otherwise the model is loaded from the cache only, and a missing-model error is shown.

Optional description JSON maps each candidate to nonempty text prompts:

```json
{"cat": ["a photo of a cat with pointed ears", "a resting domestic cat"],
 "dog": ["a photo of a dog", "a dog with a muzzle and four paws"]}
```

Independent reference images enable candidate refinement and class-specific visual fusion. The same decoded pixels cannot be supplied as both query and reference. Duplicate references are removed. Reference augmentations are seeded crop/flip views; they never use the query. Exact duplicate detection cannot identify every near-duplicate or related image: prepare genuinely separate data for evaluation. Missing descriptions use simple class-name templates, not an LLM-generated context claim.

The **CNN lab** explicitly trains the supervised CIFAR-10 classifier. Choose a new checkpoint path and enable dataset download only if needed. Reduced channel widths support CPU experimentation; width 64 retains the original channel sizes. Training uses a seeded 90/10 partition of the official training set for training and checkpoint selection, then evaluates the selected model on the official test set. It does not align CNN features to arbitrary text or support classes outside CIFAR-10. Use Streamlit's Stop control to interrupt work. Existing checkpoints are preserved.

The CLI exposes the same model operations:

```powershell
python -m vision_workbench.cnn --help
python -m vision_workbench.cnn train --help
python -m vision_workbench.cnn predict --help
```

## Equations and evaluation

`notebooks/recognition_walkthrough.ipynb` is an executed, reproducible notebook using synthetic vectors. It explains normalized contextual embeddings, independent reference fitting, class prototypes, fusion, query ranking, and exact-match evaluation. Run it from the repository root or notebooks directory using the project environment; Jupyter is optional and not installed by the application requirements. Regenerate it with `python -m vision_workbench.notebook`. Its synthetic outputs are not image-classification measurements.

The evaluation tab accepts expected and predicted labels and returns exact accuracy and a confusion matrix. It does not implement the paper's clustering permutation alignment or semantic accuracy. A lower temperature can increase the displayed top score without improving accuracy. Top-k rows retain their share of the complete retained-candidate softmax; displayed rows need not sum to one when truncated.

Automatic VQA/LLM class discovery is not implemented. Externally generated descriptions can be imported, but candidate-supplied recognition must not be described as a complete vocabulary-free pipeline. No paper-level or full-CIFAR accuracy is promised.

## Module responsibilities and checks

- `app.py`, `vision_workbench/interface.py`, `vision_workbench/ui_helpers.py`, `.streamlit/config.toml`: local interface, image preparation, duplicate protection, and styling.
- `vision_workbench/recognition.py`: NumPy equations and exact evaluation, independent of heavyweight models.
- `vision_workbench/encoders.py`: lazy pretrained CLIP image/text encoding.
- `vision_workbench/cnn.py`: supervised residual CNN training, checkpoint publication, prediction, and CLI.
- `vision_workbench/notebook.py`, `notebooks/recognition_walkthrough.ipynb`: executed educational walkthrough.
- `tests/test_*.py`: focused regressions for numerical correctness, leakage protection, UI workflows, and training/checkpoints.

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

Original documents contain personal academic details and image rights were not established, so they are not republished. Keep datasets, personal images, checkpoints, and exported results outside Git. Model loading uses safe weight-loading paths; only use checkpoints you trust. Keep the server on localhost: this is a personal coursework application, not a hosted multi-user service.

## Verified results

Automated checks: 12 tests passed with the temporary model stack before its removal. Rechecking with the existing Conda `ml` environment passed 10 tests and skipped the two optional Torch tests. Ruff checks passed. Manual Chromium review passed at desktop, tablet and mobile sizes. CNN tests use small synthetic datasets and checkpoints, not full CIFAR-10 training. Pretrained CLIP inference and the Colab notebook were not executed successfully; model download was stopped at the user's request. No paper or real-image accuracy is claimed.

The temporary environment and partial model cache created during maintenance were removed. Existing Conda environments were not modified. Heavy execution is explicitly delegated to Colab. The Colab notebook has empty outputs by design; the separate equation walkthrough has executed synthetic outputs.

The existing `cnn` environment successfully detected the GTX 1050 Ti with TensorFlow-GPU 2.10, CUDA 11.2, and cuDNN 8.1 during a small tensor check. It remains a separate TensorFlow workflow; it is not a PyTorch environment.

The CNN trainer automatically selects CUDA when available (intended for Colab) and otherwise uses CPU. CUDA training was not executed locally.

Final lightweight checks in the existing base environment: all three interface/helper tests passed, and a tiny synthetic CPU training/checkpoint run passed after the final CUDA-selection change. No datasets were downloaded for those checks.
