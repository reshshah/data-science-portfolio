"""Load versioned models from the registry and enforce the training feature contract.

Registry layout (one immutable directory per version):

    registry/
    ├── model_v1/
    │   ├── model.pkl              # pickled sklearn pipeline
    │   ├── feature_metadata.json  # feature names/order the model was trained on
    │   └── metrics.json           # evaluation metrics at training time
    └── model_v2/
        └── ...

Validating incoming features against feature_metadata.json is what prevents
train/serve skew: the model only ever sees the features, in the order, it was
trained on — or the request is rejected loudly.
"""

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

VERSION_PATTERN = re.compile(r"^model_v(\d+)$")


@dataclass
class ModelBundle:
    version: str
    model: object
    feature_names: list
    metrics: dict


def latest_version(registry_dir: Path) -> str:
    """Highest-numbered model_vN directory in the registry."""
    registry_dir = Path(registry_dir)
    versions = [
        (int(m.group(1)), p.name)
        for p in registry_dir.iterdir()
        if p.is_dir() and (m := VERSION_PATTERN.match(p.name))
    ]
    if not versions:
        raise FileNotFoundError(f"No model_vN directories found in {registry_dir}")
    return max(versions)[1]


def load_model(registry_dir, version: str = None) -> ModelBundle:
    """Load a specific version, or the latest if none given."""
    registry_dir = Path(registry_dir)
    version = version or latest_version(registry_dir)
    vdir = registry_dir / version
    with open(vdir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    metadata = json.loads((vdir / "feature_metadata.json").read_text())
    metrics = json.loads((vdir / "metrics.json").read_text())
    return ModelBundle(
        version=version,
        model=model,
        feature_names=metadata["feature_names"],
        metrics=metrics,
    )


def validate_features(bundle: ModelBundle, features: dict) -> list:
    """Check a feature dict against the training contract; return ordered values.

    Rejects missing and unexpected features rather than silently imputing or
    dropping — in serving, silence is how skew creeps in.
    """
    expected = set(bundle.feature_names)
    got = set(features)
    missing = sorted(expected - got)
    unexpected = sorted(got - expected)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing features: {', '.join(missing)}")
        if unexpected:
            parts.append(f"unexpected features: {', '.join(unexpected)}")
        raise ValueError("; ".join(parts))
    return [float(features[name]) for name in bundle.feature_names]
