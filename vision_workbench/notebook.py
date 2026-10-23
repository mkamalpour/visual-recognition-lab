"""Create an executed explanation of the equations using synthetic vectors."""

import contextlib
import io
import json
from pathlib import Path


def build():
    sections = [
        (
            "Synthetic equation walkthrough",
            "These small vectors verify the math; they are not image recognition or benchmark accuracy. Run from the repository directory.",
            'import numpy as np\nfrom vision_workbench.recognition import build_classifier, rank, evaluate_predictions\ntexts = [np.array([[1.,0.],[1.,.1]]), np.array([[0.,1.],[.1,1.]])]\nlabels = ["horizontal", "vertical"]\nprint("Two synthetic classes with two contextual vectors each")',
        ),
        (
            "Contextual text classifier",
            "Each prompt is normalized before averaging. Without references the classifier remains text-only.",
            'text_only = build_classifier(texts)\nprint(text_only["weights"])',
        ),
        (
            "Independent references and visual prototypes",
            "References are fitted before seeing a query. Pseudo-labelled reference groups each have their own prototype.",
            'references = np.array([[1.,.2],[.2,1.]])\nfit = build_classifier(texts, references, alpha=.7)\nprint("Reference counts:", fit["counts"])\nprint("Coupled weights:", fit["weights"])',
        ),
        (
            "Held-out query ranking",
            "Relative softmax scores depend on the candidate set and temperature. They are not calibrated probabilities of correctness.",
            'query = np.array([1.,.05])\nresults = rank(query, fit, labels, top_k=2)\nprint(results)\nassert results[0]["label"] == "horizontal"',
        ),
        (
            "Exact label evaluation",
            "This is a toy exact-match example, not cACC or semantic accuracy from the paper.",
            'print(evaluate_predictions(["horizontal","vertical"], ["horizontal","vertical"], labels))',
        ),
        (
            "Real model workflows",
            "The app exposes genuine CLIP and supervised CNN workflows. Models load only on explicit action. Run python -m vision_workbench.cnn --help for training/prediction commands. Automatic paper-style VQA/LLM discovery and paper benchmark reproduction are outside this adaptation.",
            '# Start the actual interface in a terminal:\n# python -m streamlit run app.py --server.address 127.0.0.1\nprint("See README.md for model setup and verified execution limits.")',
        ),
    ]
    cells, namespace = [], {}
    for i, (title, explanation, source) in enumerate(sections):
        if i == 0:
            source = (
                "from pathlib import Path\nimport sys\nproject = Path.cwd() if (Path.cwd() / 'vision_workbench').is_dir() else Path.cwd().parent\nsys.path.insert(0, str(project))\n"
                + source
            )
        cells.append(
            dict(
                cell_type="markdown",
                id=f"markdown-{i}",
                metadata={},
                source=f"## {title}\n\n{explanation}",
            )
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, "<walkthrough>", "exec"), namespace)
        cells.append(
            dict(
                cell_type="code",
                id=f"code-{i}",
                metadata={},
                source=source,
                execution_count=i + 1,
                outputs=[
                    dict(output_type="stream", name="stdout", text=buffer.getvalue())
                ],
            )
        )
    return json.dumps(
        dict(
            cells=cells,
            metadata={
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python",
                },
                "language_info": {"name": "python"},
            },
            nbformat=4,
            nbformat_minor=5,
        ),
        indent=1,
    ).encode()


if __name__ == "__main__":
    (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "recognition_walkthrough.ipynb"
    ).write_bytes(build())
