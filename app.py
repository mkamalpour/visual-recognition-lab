"""Streamlit entry point. Run from the repository root."""

try:
    from vision_workbench.interface import main
except (ImportError, ModuleNotFoundError) as exc:
    print(
        "\nVisual Recognition Lab could not start because a required dependency is missing.\n"
        "\nUse the assigned environment and install the project requirements first:\n"
        "  conda activate base\n"
        "  python -m pip install -r requirements.txt\n"
        "\nIf you are using the managed workspace, open the project workspace to select the right interpreter.\n"
        f"Original error: {type(exc).__name__}: {exc}\n"
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
