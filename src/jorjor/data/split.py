from __future__ import annotations

from dataclasses import dataclass
import random

from .schema import DecisionPoint


@dataclass(frozen=True)
class SplitFractions:
    """Train/validation/test split fractions."""

    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class SplitManifest:
    """Hand-id split manifest."""

    train_hand_ids: tuple[str, ...]
    validation_hand_ids: tuple[str, ...]
    test_hand_ids: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        """Serialize the split manifest."""

        return {
            "train_hand_ids": list(self.train_hand_ids),
            "validation_hand_ids": list(self.validation_hand_ids),
            "test_hand_ids": list(self.test_hand_ids),
            "counts": {
                "train_hands": len(self.train_hand_ids),
                "validation_hands": len(self.validation_hand_ids),
                "test_hands": len(self.test_hand_ids),
            },
        }


@dataclass(frozen=True)
class SplitDatasets:
    """Decision points grouped by split."""

    train: tuple[DecisionPoint, ...]
    validation: tuple[DecisionPoint, ...]
    test: tuple[DecisionPoint, ...]


def build_split_manifest(
    points: tuple[DecisionPoint, ...],
    fractions: SplitFractions,
    seed: int,
) -> SplitManifest:
    """Create a deterministic split manifest by hand id."""

    total = fractions.train + fractions.validation + fractions.test
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    hand_ids = sorted({point.hand_id for point in points})
    rng = random.Random(seed)
    shuffled = hand_ids[:]
    rng.shuffle(shuffled)

    train_end = int(len(shuffled) * fractions.train)
    validation_end = train_end + int(len(shuffled) * fractions.validation)
    return SplitManifest(
        train_hand_ids=tuple(sorted(shuffled[:train_end])),
        validation_hand_ids=tuple(sorted(shuffled[train_end:validation_end])),
        test_hand_ids=tuple(sorted(shuffled[validation_end:])),
    )


def apply_split(points: tuple[DecisionPoint, ...], manifest: SplitManifest) -> SplitDatasets:
    """Split decision points using a hand-id manifest."""

    train_ids = set(manifest.train_hand_ids)
    validation_ids = set(manifest.validation_hand_ids)
    test_ids = set(manifest.test_hand_ids)
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("split manifest contains overlapping hand ids")

    train: list[DecisionPoint] = []
    validation: list[DecisionPoint] = []
    test: list[DecisionPoint] = []
    for point in points:
        if point.hand_id in train_ids:
            train.append(point)
        elif point.hand_id in validation_ids:
            validation.append(point)
        elif point.hand_id in test_ids:
            test.append(point)
        else:
            raise ValueError(f"point hand_id is missing from split manifest: {point.hand_id}")

    return SplitDatasets(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
    )


def has_split_overlap(manifest: SplitManifest) -> bool:
    """Return True if any hand id appears in multiple splits."""

    train_ids = set(manifest.train_hand_ids)
    validation_ids = set(manifest.validation_hand_ids)
    test_ids = set(manifest.test_hand_ids)
    return bool(train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids)
