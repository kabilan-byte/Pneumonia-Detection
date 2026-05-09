"""
src/train.py — Training Script
================================
End-to-end training pipeline:
    1. Build the model (CNN or Transfer Learning)
    2. Compile with Adam optimizer and binary cross-entropy loss
    3. Train with smart callbacks (EarlyStopping, ReduceLROnPlateau, Checkpoint)
    4. Plot and save training curves
    5. Save the final model

Usage:
    python src/train.py                          # Uses config.py defaults
    python src/train.py --model mobilenetv2      # Override model type
    python src/train.py --model resnet50 --epochs 40
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_TYPE, MODEL_SAVE_PATH, FINAL_MODEL_PATH, RESULTS_DIR,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, FINE_TUNE_LR,
    EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR, MIN_LR,
    INPUT_SHAPE
)
# Support both:  python src/train.py  AND  python -m src.train
try:
    from src.augmentation import get_train_generator, get_val_generator
    from src.model_cnn import build_custom_cnn
    from src.model_transfer import get_transfer_model, unfreeze_top_layers
except ModuleNotFoundError:
    from augmentation import get_train_generator, get_val_generator
    from model_cnn import build_custom_cnn
    from model_transfer import get_transfer_model, unfreeze_top_layers

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _make_callbacks(model_path: str, log_dir: str) -> list:
    """Return a list of standard training callbacks."""
    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]
    # TensorBoard callback — only add if tensorboard is installed
    try:
        import tensorboard  # noqa
        os.makedirs(log_dir, exist_ok=True)
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
    except ImportError:
        print("ℹ️  TensorBoard not installed — skipping TB callback.")
    return callbacks


def _plot_history(history, save_dir: str, tag: str = ""):
    """Plot and save accuracy + loss curves from training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=15, fontweight="bold")

    # ── Accuracy ────────────────────────────────────────────────
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Loss ────────────────────────────────────────────────────
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"training_curves{tag}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Training curves saved → {fname}")


# ──────────────────────────────────────────────────────────────
# CORE TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────

def train_model(model_type: str = MODEL_TYPE,
                epochs: int = EPOCHS,
                batch_size: int = BATCH_SIZE,
                learning_rate: float = LEARNING_RATE,
                fine_tune: bool = True,
                fine_tune_epochs: int = 10,
                fine_tune_layers: int = 30):
    """
    Full training pipeline.

    Phase 1: Train custom head only (base frozen for transfer models).
    Phase 2: Unfreeze top layers and fine-tune with lower LR.

    Args:
        model_type       (str):   "cnn" | "mobilenetv2" | "resnet50" | "efficientnetb0"
        epochs           (int):   Max epochs for Phase 1.
        batch_size       (int):   Batch size.
        learning_rate    (float): Phase 1 learning rate.
        fine_tune        (bool):  Run Phase 2 fine-tuning (transfer models only).
        fine_tune_epochs (int):   Max epochs for Phase 2.
        fine_tune_layers (int):   Number of base layers to unfreeze.

    Returns:
        tf.keras.Model: The trained model.
    """
    print("\n" + "=" * 60)
    print(f"  🫁 Pneumonia Detection Training")
    print(f"     Model     : {model_type.upper()}")
    print(f"     Epochs    : {epochs} (Phase 1) + {fine_tune_epochs if fine_tune else 0} (Phase 2)")
    print(f"     Batch Size: {batch_size}")
    print(f"     LR        : {learning_rate}")
    print("=" * 60 + "\n")

    # ── Data generators ──────────────────────────────────────────
    train_gen = get_train_generator(batch_size=batch_size)
    val_gen   = get_val_generator(batch_size=batch_size)

    # ── Build model ──────────────────────────────────────────────
    if model_type.lower() == "cnn":
        model = build_custom_cnn(input_shape=INPUT_SHAPE)
        fine_tune = False   # No base to unfreeze for custom CNN
    else:
        model = get_transfer_model(model_type, input_shape=INPUT_SHAPE)

    model.summary()

    # ── Phase 1 — Train head ─────────────────────────────────────
    print("\n🔵 Phase 1: Training classification head (base frozen)...")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    log_dir = os.path.join(RESULTS_DIR, "logs", "phase1")
    callbacks = _make_callbacks(MODEL_SAVE_PATH, log_dir)

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )
    _plot_history(history1, RESULTS_DIR, tag="_phase1")

    # ── Phase 2 — Fine-tune ──────────────────────────────────────
    if fine_tune:
        print(f"\n🟠 Phase 2: Fine-tuning top {fine_tune_layers} base layers...")
        model = unfreeze_top_layers(model, num_layers=fine_tune_layers)

        model.compile(
            optimizer=Adam(learning_rate=FINE_TUNE_LR),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        log_dir2 = os.path.join(RESULTS_DIR, "logs", "phase2")
        callbacks2 = _make_callbacks(MODEL_SAVE_PATH, log_dir2)

        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=fine_tune_epochs,
            callbacks=callbacks2,
        )
        _plot_history(history2, RESULTS_DIR, tag="_phase2")

    # ── Save final model ─────────────────────────────────────────
    model.save(FINAL_MODEL_PATH)
    print(f"\n✅ Final model saved → {FINAL_MODEL_PATH}")
    print(f"✅ Best model saved  → {MODEL_SAVE_PATH}")

    return model


# ──────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pneumonia Detection model")
    parser.add_argument("--model",           type=str,   default=MODEL_TYPE,
                        choices=["cnn", "mobilenetv2", "resnet50", "efficientnetb0"],
                        help="Model architecture to use")
    parser.add_argument("--epochs",          type=int,   default=EPOCHS,
                        help="Number of training epochs (Phase 1)")
    parser.add_argument("--batch-size",      type=int,   default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--lr",              type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--fine-tune",       action="store_true", default=True,
                        help="Enable Phase 2 fine-tuning (transfer models only)")
    parser.add_argument("--no-fine-tune",    action="store_false", dest="fine_tune",
                        help="Skip Phase 2 fine-tuning")
    parser.add_argument("--fine-tune-epochs",type=int,   default=10,
                        help="Epochs for Phase 2 fine-tuning")
    parser.add_argument("--fine-tune-layers",type=int,   default=30,
                        help="Number of base layers to unfreeze in Phase 2")

    args = parser.parse_args()

    trained_model = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_layers=args.fine_tune_layers,
    )
