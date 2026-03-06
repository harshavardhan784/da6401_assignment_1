"""
Hyperparameter Sweep - Q2.2
Bayesian sweep, 100 runs targeting maximum validation accuracy.

After the sweep, go to W&B -> Sweeps -> your sweep -> "Parallel Coordinates" plot
to identify which hyperparameter had the most significant impact on val_acc.

Usage (from da6401_assignment_1/src/):
    python -m sweep --wandb_project da6401-a1 --count 100
    python -m sweep --wandb_project da6401-a1 --wandb_entity <your-entity> --count 100
"""

import argparse
import types
import numpy as np
import wandb

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.objective_functions import compute_loss
from ann.optimizers import NAG
from ann.activations import softmax

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1,
        },
        "batch_size":   {"values": [32, 64, 128]},
        "num_layers":   {"values": [2, 3, 4]},
        "hidden_size":  {"values": [64, 128]},
        "activation":   {"values": ["relu", "sigmoid", "tanh"]},
        "optimizer":    {"values": ["sgd", "momentum", "nag",
                                    "rmsprop", "adam", "nadam"]},
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "weight_init":  {"values": ["xavier"]},    # random poisons Bayesian search
        "loss":         {"values": ["cross_entropy"]},
        "epochs":       {"value": 6},
    },
}


# Load data once — shared across all, sweep runs (avoids repeated disk I/O)
print("Pre-loading MNIST dataset for sweep...")
X_train, y_train, X_val, y_val, X_test, y_test = load_data("mnist")
print(f"  train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")


def sweep_agent_fn():
    run = wandb.init(
        settings=wandb.Settings(
            _disable_stats=True,
            _disable_meta=True,
            save_code=False,
            _service_wait=120,
        )
    )
    cfg = run.config

    try:
        args = types.SimpleNamespace(
            dataset       = "mnist",
            epochs        = cfg.epochs,
            batch_size    = cfg.batch_size,
            loss          = cfg.loss,
            optimizer     = cfg.optimizer,
            learning_rate = cfg.learning_rate,
            weight_decay  = cfg.weight_decay,
            num_layers    = cfg.num_layers,
            hidden_size   = [cfg.hidden_size] * cfg.num_layers,
            activation    = cfg.activation,
            weight_init   = cfg.weight_init,
            input_dim     = 784,
            output_dim    = 10,
        )

        model     = NeuralNetwork(args)
        is_nag    = isinstance(model.optimizer, NAG)
        n_samples = X_train.shape[0]
        best_val  = -1.0

        for epoch in range(args.epochs):
            idx  = np.random.permutation(n_samples)
            X_tr = X_train[idx]
            y_tr = y_train[idx]

            epoch_loss  = 0.0
            num_batches = 0

            for start in range(0, n_samples, args.batch_size):
                Xb = X_tr[start : start + args.batch_size]
                yb = y_tr[start : start + args.batch_size]

                if is_nag:
                    model.optimizer.apply_lookahead(model.layers)
                    logits = model.forward(Xb)
                    probs  = softmax(logits)
                    bl     = compute_loss(yb, probs, args.loss)
                    model.backward(yb, probs)   # pass PROBS
                    model.optimizer.restore_weights(model.layers)
                else:
                    logits = model.forward(Xb)
                    probs  = softmax(logits)
                    bl     = compute_loss(yb, probs, args.loss)
                    model.backward(yb, probs)   # pass PROBS

                model.update_weights()
                epoch_loss  += bl
                num_batches += 1

            avg_loss  = epoch_loss / num_batches
            val_acc   = model.evaluate(X_val, y_val)
            train_acc = model.evaluate(X_tr, y_tr)
            val_probs = softmax(model.forward(X_val))
            val_loss  = compute_loss(y_val, val_probs, args.loss)

            wandb.log({
                "epoch":      epoch + 1,
                "train_loss": float(avg_loss),
                "val_loss":   float(val_loss),
                "train_acc":  float(train_acc),
                "val_acc":    float(val_acc),
            })

            if val_acc > best_val:
                best_val = val_acc

        test_acc = model.evaluate(X_test, y_test)
        run.summary["best_val_acc"] = best_val
        run.summary["test_acc"]     = test_acc

    except Exception as exc:
        print(f"[sweep] Run failed: {exc}")
        wandb.log({"run_failed": 1})

    finally:
        run.finish()     # always close — ensures next run starts cleanly


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Q2.2 — Hyperparameter Sweep")
    parser.add_argument("--wandb_project", type=str, default="da6401-a1")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument("--count",         type=int, default=100,
                        help="Number of sweep runs")
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        SWEEP_CONFIG,
        project=args.wandb_project,
        entity =args.wandb_entity,
    )
    print(f"Sweep ID : {sweep_id}")
    print(f"View at  : https://wandb.ai/"
          f"{args.wandb_entity or '<entity>'}/"
          f"{args.wandb_project}/sweeps/{sweep_id}")
    print(f"Starting {args.count} sweep runs...")

    wandb.agent(
        sweep_id,
        function=sweep_agent_fn,
        count   =args.count,
        project =args.wandb_project,
        entity  =args.wandb_entity,
    )
    print("All sweep runs complete.")


if __name__ == "__main__":
    main()