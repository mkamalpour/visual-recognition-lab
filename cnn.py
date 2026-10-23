"""Supervised residual CNN. Its features are not CLIP text embeddings."""

from pathlib import Path

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_model(width=64):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Install requirements-models.txt in the project environment to use the CNN."
        ) from error
    from torch import nn

    if not isinstance(width, int) or not 2 <= width <= 64:
        raise ValueError("Width must be an integer from 2 to 64.")

    class Block(nn.Module):
        def __init__(self, a, b, stride=1):
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(a, b, 3, stride, 1, bias=False),
                nn.BatchNorm2d(b),
                nn.GELU(),
                nn.Conv2d(b, b, 3, 1, 1, bias=False),
                nn.BatchNorm2d(b),
            )
            self.skip = (
                nn.Identity()
                if a == b and stride == 1
                else nn.Sequential(
                    nn.Conv2d(a, b, 1, stride, bias=False), nn.BatchNorm2d(b)
                )
            )

        def forward(self, x):
            return torch.nn.functional.gelu(self.main(x) + self.skip(x))

    layers = [
        nn.Conv2d(3, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width),
        nn.GELU(),
    ]
    a = width
    for b, stride in [(width * 2, 2), (width * 4, 2), (width * 8, 2), (width * 8, 1)]:
        layers.extend([Block(a, b, stride), Block(b, b)])
        a = b
    layers.extend(
        [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(a, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.Linear(512, 10),
        ]
    )
    return nn.Sequential(*layers)


def transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )


def train_cifar10(
    data_dir,
    checkpoint,
    epochs=15,
    batch_size=128,
    download=False,
    width=64,
    progress=None,
):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Install requirements-models.txt in the project environment to use the CNN."
        ) from error
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import CIFAR10

    if (
        not isinstance(epochs, int)
        or epochs < 1
        or not isinstance(batch_size, int)
        or batch_size < 2
    ):
        raise ValueError("Use positive epochs and batch size at least 2.")
    target = Path(checkpoint)
    if target.exists():
        raise FileExistsError(
            "Choose a new checkpoint path; existing work is preserved."
        )
    torch.manual_seed(42)
    dataset = CIFAR10(data_dir, train=True, download=download, transform=transform())
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(42)
    ).tolist()
    split = int(len(indices) * 0.9)
    train = DataLoader(
        Subset(dataset, indices[:split]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid = DataLoader(Subset(dataset, indices[split:]), batch_size=batch_size)
    test = DataLoader(
        CIFAR10(data_dir, train=False, download=download, transform=transform()),
        batch_size=batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 12], gamma=0.1
    )

    def accuracy(loader):
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        return correct / total

    best = -1
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        for batch, (x, y) in enumerate(train):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            if progress:
                progress(
                    (epoch + (batch + 1) / len(train)) / epochs,
                    f"Epoch {epoch + 1}/{epochs}",
                )
        score = accuracy(valid)
        history.append({"epoch": epoch + 1, "validation_accuracy": score})
        if score > best:
            best = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        scheduler.step()
    model.load_state_dict(best_state)
    test_accuracy = accuracy(test)
    target.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(
                {
                    "state": best_state,
                    "width": width,
                    "classes": CLASSES,
                    "validation_accuracy": best,
                    "test_accuracy": test_accuracy,
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        # Linking publishes only a complete file and fails if destination exists.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "history": history,
        "validation_accuracy": best,
        "test_accuracy": test_accuracy,
        "checkpoint": str(target),
    }


def predict_image(image, checkpoint):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Install requirements-models.txt in the project environment to use the CNN."
        ) from error

    if not Path(checkpoint).is_file():
        raise FileNotFoundError("Select an existing trained checkpoint file.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("classes") != CLASSES:
        raise ValueError("Not a supported CIFAR-10 checkpoint.")
    model = build_model(payload["width"])
    model.load_state_dict(payload["state"])
    model.eval()
    with torch.inference_mode():
        scores = (
            model(transform()(image.convert("RGB")).unsqueeze(0)).softmax(1)[0].tolist()
        )
    return sorted(
        [{"label": label, "score": score} for label, score in zip(CLASSES, scores)],
        key=lambda row: -row["score"],
    )


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="Train on CIFAR-10 with separate validation/test sets"
    )
    train.add_argument("--data-dir", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--width", type=int, default=64)
    train.add_argument("--download", action="store_true")
    predict = commands.add_parser(
        "predict", help="Classify using a trained CIFAR-10 checkpoint"
    )
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "train":
        result = train_cifar10(**args)
    else:
        from PIL import Image

        with Image.open(args["image"]) as image:
            result = predict_image(image, args["checkpoint"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
