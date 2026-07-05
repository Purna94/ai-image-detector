import os
import shutil
import argparse
from pathlib import Path

import torch
from datasets import Dataset, Image as HFImage, Features, ClassLabel, Value
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
    ToTensor,
)
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
DEFAULT_OUTPUT = BASE_DIR / "model_output"
DEFAULT_HF_REPO = "Purna94/ai-image-detector"

ID2LABEL = {0: "REAL", 1: "FAKE"}
LABEL2ID = {"REAL": 0, "FAKE": 1}


def load_images_from_folder(folder: Path, label: int, max_samples: int | None = None):
    paths = sorted(folder.rglob("*"))
    files = [p for p in paths if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    if max_samples:
        files = files[:max_samples]
    return [{"image": str(f), "label": label} for f in files]


def create_dataset(
    train_fake: Path,
    train_real: Path,
    test_fake: Path,
    test_real: Path,
    max_train: int | None = None,
    max_test: int | None = None,
):
    train_data = load_images_from_folder(train_fake, LABEL2ID["FAKE"], max_train)
    train_data += load_images_from_folder(train_real, LABEL2ID["REAL"], max_train)
    test_data = load_images_from_folder(test_fake, LABEL2ID["FAKE"], max_test)
    test_data += load_images_from_folder(test_real, LABEL2ID["REAL"], max_test)

    features = Features({"image": Value("string"), "label": ClassLabel(num_classes=2, names=["REAL", "FAKE"])})

    train_ds = Dataset.from_list(train_data, features=features).cast_column("image", HFImage())
    test_ds = Dataset.from_list(test_data, features=features).cast_column("image", HFImage())
    return train_ds, test_ds


def transform_images(examples, processor):
    transform = Compose(
        [
            Resize((processor.size["height"], processor.size["width"])),
            CenterCrop(processor.size["height"]),
            ToTensor(),
            Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    def apply(image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return transform(image)

    examples["pixel_values"] = [apply(img) for img in examples["image"]]
    return examples


def transform_train(examples, processor):
    transform = Compose(
        [
            Resize((processor.size["height"], processor.size["width"])),
            RandomRotation(10),
            RandomHorizontalFlip(),
            CenterCrop(processor.size["height"]),
            ToTensor(),
            Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    def apply(image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return transform(image)

    examples["pixel_values"] = [apply(img) for img in examples["image"]]
    return examples


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = (predictions == labels).mean()
    return {"accuracy": acc}


def main():
    parser = argparse.ArgumentParser(description="Train AI Image Detector")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--hf_repo", type=str, default=DEFAULT_HF_REPO)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, test_ds = create_dataset(
        TRAIN_DIR / "FAKE",
        TRAIN_DIR / "REAL",
        TEST_DIR / "FAKE",
        TEST_DIR / "REAL",
        args.max_train,
        args.max_test,
    )
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    processor = ViTImageProcessor.from_pretrained(args.model_name)
    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_ds.set_transform(lambda x: transform_train(x, processor))
    test_ds.set_transform(lambda x: transform_images(x, processor))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hf_repo if args.push_to_hub else None,
        hub_strategy="end",
        fp16=args.fp16 and device.type == "cuda",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    if args.push_to_hub:
        trainer.push_to_hub()
        print(f"Model pushed to HuggingFace Hub: {args.hf_repo}")

    eval_result = trainer.evaluate(test_ds)
    print(f"Evaluation results: {eval_result}")


if __name__ == "__main__":
    main()
