"""Local visual recognition research interface; optional models load on request."""

import hashlib
import json
from pathlib import Path

import numpy as np
import streamlit as st

from vision_workbench.recognition import build_classifier, evaluate_predictions, rank
from vision_workbench.ui_helpers import (
    decode_image,
    image_variants,
    reference_images,
    validate_descriptions,
)


def main():
    st.set_page_config(page_title="Visual Recognition Lab", layout="wide")
    st.markdown(
        """<style>.stApp{background:#f6f3ec;color:#202b2e}.block-container{padding-top:3.5rem;max-width:1200px}h1{font-family:Georgia,serif;font-size:3rem!important;letter-spacing:-1px}h2,h3{color:#176963}button[kind=primary]{background:#176963}div[data-testid=stMetric]{border-top:3px solid #d97859;padding:12px 0}div[data-baseweb=tab-list]{gap:18px;overflow-x:auto}button[data-baseweb=tab]{white-space:nowrap}@media(max-width:600px){h1{font-size:2rem!important}.block-container{padding:3rem 1rem 1rem}}</style>""",
        unsafe_allow_html=True,
    )
    st.caption("VISION / RESEARCH STUDIO")
    st.title("Visual Recognition Lab")
    st.write(
        "Explore image recognition, inspect the method, and train a small convolutional network."
    )
    overview, clip, cnn_tab, evaluation = st.tabs(
        ["Start here", "CLIP recognition", "CNN lab", "Evaluation & method"]
    )
    with overview:
        st.header("From an image to an inspectable decision")
        st.write(
            "Use CLIP to compare an image against candidate descriptions, or train the four-stage CNN on CIFAR-10."
        )
        a, b, c = st.columns(3)
        a.metric("Recognition paths", "2")
        b.metric("CNN classes", "10")
        c.metric("Runs on", "Your machine")
        st.subheader("Choose your workflow")
        st.markdown(
            "1. **CLIP recognition** — upload a query, choose candidate names, and optionally add independent reference images.\n2. **CNN lab** — explicitly download CIFAR-10, train a checkpoint, then predict an image.\n3. **Evaluation & method** — check labelled predictions and inspect a synthetic vector example."
        )
        st.info(
            "CLIP requires optional model dependencies and pretrained weights. Nothing downloads or trains when this app opens. This is an educational adaptation, not a reproduction of E-FineR benchmark results."
        )
    with clip:
        st.header("Compare an image with candidate descriptions")
        st.caption(
            "Scores are relative softmax scores over your candidates, not calibrated confidence. The query is never used to fit the classifier."
        )
        left, right = st.columns([1, 1])
        with left:
            query = st.file_uploader("Query image", type=["png", "jpg", "jpeg", "webp"])
            domain = st.selectbox(
                "Candidate starter", ["Custom", "Animals", "Birds", "Cars", "Fruits"]
            )
            presets = {
                "Custom": "cat\ndog\nbird",
                "Animals": "cat\ndog\nrabbit\nmouse\nelephant\nlion\ntiger\nbear\nfox\nwolf",
                "Birds": "cardinal\nblue jay\nrobin\neagle\nhawk\nsparrow\ncrow\nowl\nparrot\npigeon",
                "Cars": "ferrari\nporsche\ntesla\nbmw\nmercedes\naudi\ntoyota\nford\nhonda\nvolvo",
                "Fruits": "apple\norange\nbanana\ngrape\nstrawberry\nwatermelon\nmango\npeach\npear\ncherry",
            }
            candidates = st.text_area(
                "One candidate name per line", presets[domain], key="names_" + domain
            )
            descriptions_file = st.file_uploader(
                "Candidate descriptions JSON (optional)", type=["json"]
            )
            refs = st.file_uploader(
                "Reference images (optional; separate from query)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
            )
        with right:
            if query:
                try:
                    st.image(
                        decode_image(query.getvalue()),
                        caption="Query image",
                        use_container_width=True,
                    )
                except Exception:
                    st.error("This file cannot be decoded as an image.")
            else:
                st.info("Upload a query image to begin. Reference images are optional.")
            with st.expander("Model & refinement controls"):
                allow = st.checkbox("Allow pretrained model download", False)
                augmentations = st.number_input("Reference views per image", 1, 8, 1)
                alpha = st.slider("Text weight", 0.0, 1.0, 0.7)
                temperature = st.slider("Score temperature", 0.01, 1.0, 0.07)
                keep = st.number_input("Maximum retained candidates", 1, 1000, 100)
                topk = st.number_input("Results to show", 1, 100, 3)
        fingerprint = hashlib.sha256(
            repr(
                (
                    query.getvalue() if query else None,
                    [r.getvalue() for r in refs],
                    candidates,
                    descriptions_file.getvalue() if descriptions_file else None,
                    augmentations,
                    alpha,
                    temperature,
                    keep,
                    topk,
                )
            ).encode()
        ).hexdigest()
        if st.button("Recognize image", type="primary", disabled=query is None):
            try:
                names = [x.strip() for x in candidates.splitlines() if x.strip()]
                if len(names) > 1000 or len(refs) > 100:
                    raise ValueError(
                        "Use at most 1000 candidates and 100 reference images."
                    )
                if not names or len(set(names)) != len(names):
                    raise ValueError("Provide unique, nonempty candidate names.")
                reference_pictures = reference_images(
                    query.getvalue(), [r.getvalue() for r in refs]
                )
                prompts_per_class = validate_descriptions(
                    names,
                    json.loads(descriptions_file.getvalue())
                    if descriptions_file
                    else {},
                )
                from vision_workbench.encoders import CLIPEncoder

                with st.spinner("Encoding images and candidate descriptions..."):
                    encoder = CLIPEncoder(allow_download=allow)
                    texts = [
                        encoder.encode_texts(prompts) for prompts in prompts_per_class
                    ]
                    reference = (
                        encoder.encode_images(reference_pictures) if refs else None
                    )
                    augmented = (
                        np.stack(
                            [
                                encoder.encode_images(
                                    image_variants(picture, int(augmentations))
                                )
                                for picture in reference_pictures
                            ]
                        )
                        if reference_pictures
                        else None
                    )
                    classifier = build_classifier(
                        texts,
                        reference_embeddings=reference,
                        augmented_embeddings=augmented,
                        alpha=alpha,
                        keep=min(int(keep), len(names)) if refs else len(names),
                    )
                    result = rank(
                        encoder.encode_images([decode_image(query.getvalue())]),
                        classifier,
                        names,
                        top_k=min(int(topk), len(classifier["indices"])),
                        temperature=temperature,
                    )
                st.session_state.recognition_result = (fingerprint, result)
            except Exception as exc:
                st.error(str(exc))
        saved = st.session_state.get("recognition_result")
        if saved and saved[0] == fingerprint:
            st.subheader("Ranked candidates")
            if saved[1]:
                st.write(
                    f"Top candidate: **{saved[1][0]['label']}** ? relative score {saved[1][0]['score']:.1%}"
                )
                st.bar_chart(saved[1], x="label", y="score", horizontal=True)
            st.dataframe(saved[1], hide_index=True, use_container_width=True)
            st.download_button(
                "Download recognition JSON",
                json.dumps(saved[1], indent=2),
                "recognition.json",
                "application/json",
            )
        elif saved:
            st.info("Inputs changed. Run recognition again for current results.")
    with cnn_tab:
        st.header("Train and inspect a CIFAR-10 CNN")
        st.write(
            "Training is a real compute-intensive operation. Use the Streamlit Stop control to interrupt a run."
        )
        data = st.text_input("Local CIFAR-10 data directory", "data/cifar10")
        checkpoint = st.text_input("New checkpoint path", "outputs/cnn.pt")
        epochs = st.number_input("Epochs", 1, 1000, 5)
        batch = st.selectbox("Batch size", [32, 64, 128], index=1)
        width = st.select_slider(
            "CNN channel width", options=[2, 4, 8, 16, 32, 64], value=16
        )
        download = st.checkbox("Download CIFAR-10 if missing")
        training_inputs = {
            "data_directory": data, "checkpoint": checkpoint,
            "epochs": int(epochs), "batch_size": batch, "channel_width": width,
            "download_allowed": download,
        }
        if st.button("Train CNN", type="primary"):
            try:
                from vision_workbench.cnn import train_cifar10

                with st.spinner("Training CNN…"):
                    status = st.progress(0.0, text="Preparing dataset")
                    result = train_cifar10(
                        data,
                        checkpoint,
                        epochs=int(epochs),
                        batch_size=batch,
                        width=width,
                        download=download,
                        progress=lambda fraction, message: status.progress(
                            fraction, text=message
                        ),
                    )
                st.session_state.training_result = result
                st.session_state.completed_training_inputs = training_inputs.copy()
            except Exception as exc:
                st.error(str(exc))
        if "training_result" in st.session_state:
            recorded = st.session_state.get("completed_training_inputs")
            st.subheader("Last completed training run")
            if recorded != training_inputs:
                st.info("Training settings changed. The report below belongs to the previous completed run, not the current controls. Train again to produce a new report.")
            if recorded:
                with st.expander("Settings used for this report"):
                    st.json(recorded)
            st.json(st.session_state.training_result)
            st.download_button(
                "Download training metrics",
                json.dumps(st.session_state.training_result, indent=2, default=str),
                "training.json",
                "application/json",
            )
        st.divider()
        st.subheader("Predict with a saved checkpoint")
        image = st.file_uploader(
            "CNN query image", type=["png", "jpg", "jpeg"], key="cnn_image"
        )
        if image:
            try:
                st.image(
                    decode_image(image.getvalue()), caption="CNN query image", width=320
                )
            except Exception as exc:
                st.error(str(exc))
        checkpoint_file = Path(checkpoint)
        checkpoint_stamp = (
            (checkpoint_file.stat().st_size, checkpoint_file.stat().st_mtime_ns)
            if checkpoint_file.is_file()
            else None
        )
        cnn_fingerprint = hashlib.sha256(
            repr(
                (checkpoint, checkpoint_stamp, image.getvalue() if image else None)
            ).encode()
        ).hexdigest()
        if st.button("Predict with CNN", disabled=image is None):
            try:
                from vision_workbench.cnn import predict_image

                st.session_state.cnn_result = (
                    cnn_fingerprint,
                    predict_image(decode_image(image.getvalue()), checkpoint),
                )
            except Exception as exc:
                st.error(str(exc))
        cnn_saved = st.session_state.get("cnn_result")
        if cnn_saved and cnn_saved[0] == cnn_fingerprint:
            if cnn_saved[1]:
                st.write(
                    f"Top CIFAR-10 class: **{cnn_saved[1][0]['label']}** ? relative score {cnn_saved[1][0]['score']:.1%}"
                )
                st.caption("Relative softmax scores are not calibrated confidence.")
                st.bar_chart(cnn_saved[1], x="label", y="score", horizontal=True)
            st.dataframe(cnn_saved[1], hide_index=True)
        elif cnn_saved:
            st.info(
                "CNN inputs or checkpoint changed. Predict again for current results."
            )
    with evaluation:
        st.header("Measure the task you actually ran")
        st.write(
            "Exact label accuracy measures classification. Clustering accuracy requires permutation alignment and is a different metric. Neither is a semantic similarity score."
        )
        expected = st.text_area("Expected labels, one per line", "cat\ndog\ncat")
        predicted = st.text_area("Predicted labels, one per line", "cat\ncat\ncat")
        evaluation_key = (expected, predicted)
        if st.button("Evaluate labelled predictions"):
            try:
                truth = [line.strip() for line in expected.splitlines()]
                pred = [line.strip() for line in predicted.splitlines()]
                if any(not label for label in truth + pred):
                    raise ValueError('Remove blank label rows; each row must identify one sample.')
                result = evaluate_predictions(truth, pred, sorted(set(truth + pred)))
                st.session_state.evaluation_result = (evaluation_key, result)
            except Exception as exc:
                st.error(str(exc))
        evaluation_saved = st.session_state.get('evaluation_result')
        if evaluation_saved and evaluation_saved[0] == evaluation_key:
            result = evaluation_saved[1]
            a, b, c = st.columns(3)
            a.metric('Exact accuracy', f"{result['accuracy']:.1%}")
            b.metric('Balanced accuracy', f"{result['balanced_accuracy']:.1%}")
            c.metric('Macro F1', f"{result['macro_f1']:.1%}")
            st.caption(result['metric_policy'])
            st.subheader('Per-class results')
            st.dataframe(result['per_class'], hide_index=True)
            st.subheader('Confused label pairs')
            if result['errors']:
                st.dataframe(result['errors'], hide_index=True)
            else:
                st.success('All supplied predictions match their expected labels.')
            st.download_button('Download evaluation report', json.dumps(result, indent=2),
                               'recognition-evaluation.json', 'application/json')
            with st.expander('Confusion matrix (rows = expected, columns = predicted)'):
                st.json(dict(labels=result['labels'], confusion=result['confusion']))
        elif evaluation_saved:
            st.info('Evaluation inputs changed. Evaluate again to refresh the report and export.')
        with st.expander("Inspect a synthetic vector example"):
            st.caption(
                "Synthetic vectors only: this does not recognize images or measure model quality."
            )
            model = build_classifier([np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])])
            st.dataframe(
                rank(np.array([0.9, 0.1]), model, ["horizontal", "vertical"]),
                hide_index=True,
            )
        st.subheader("What this implementation does")
        st.write(
            "Normalize embeddings → average candidate prompts → optionally refine candidates using independent references → build class-specific pseudo-labelled visual prototypes → blend text and visual weights → rank a held-out query."
        )
        st.warning(
            "Candidate names are supplied by you. Automatic vocabulary discovery and published E-FineR accuracy are not claimed. Never use a query image as its own reference."
        )
