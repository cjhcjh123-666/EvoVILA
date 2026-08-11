"""Metadata-only contracts required before EvoVILA segmentation training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Tuple


MediaType = Literal["image", "video"]
ControlKind = Literal["positive", "no_object", "empty_query"]
DataSplit = Literal["train", "val", "test"]
RetentionTask = Literal["text", "single_image", "multi_image", "video"]
RetentionComparison = Literal["exact_logits", "exact_tokens", "task_metric"]


def _validate_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string")
    return value


def _validate_absolute_path(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string")
    if not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute external path")
    return value


def _counter_mapping(values: Any) -> Mapping[str, int]:
    return MappingProxyType(dict(sorted(Counter(values).items())))


@dataclass(frozen=True)
class MaskTrainingRecord:
    """One positive target or negative control without opening external data."""

    sample_id: str
    media_id: str
    media_type: MediaType
    media_path: str
    frame_indices: Tuple[int, ...]
    mask_paths: Tuple[Optional[str], ...]
    target_presence: Tuple[bool, ...]
    anchor_position: int
    split: DataSplit
    query: str
    target_id: Optional[str]
    control_kind: ControlKind = "positive"

    def __post_init__(self) -> None:
        _validate_identifier("sample_id", self.sample_id)
        _validate_identifier("media_id", self.media_id)
        _validate_absolute_path("media_path", self.media_path)
        if self.media_type not in ("image", "video"):
            raise ValueError("media_type must be 'image' or 'video'")
        if self.split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")
        if self.control_kind not in ("positive", "no_object", "empty_query"):
            raise ValueError("control_kind must be positive, no_object, or empty_query")
        if not isinstance(self.frame_indices, tuple) or not self.frame_indices:
            raise TypeError("frame_indices must be a non-empty tuple")
        if not all(isinstance(index, int) and not isinstance(index, bool) for index in self.frame_indices):
            raise TypeError("frame_indices must contain integers")
        if any(index < 0 for index in self.frame_indices):
            raise ValueError("frame_indices must be non-negative")
        if any(left >= right for left, right in zip(self.frame_indices, self.frame_indices[1:])):
            raise ValueError("frame_indices must be strictly increasing")
        frame_count = len(self.frame_indices)
        if not isinstance(self.mask_paths, tuple) or len(self.mask_paths) != frame_count:
            raise ValueError("mask_paths must be a tuple aligned with frame_indices")
        if not isinstance(self.target_presence, tuple) or len(self.target_presence) != frame_count:
            raise ValueError("target_presence must be a tuple aligned with frame_indices")
        if not all(type(value) is bool for value in self.target_presence):
            raise TypeError("target_presence must contain bool values")
        for index, mask_path in enumerate(self.mask_paths):
            if mask_path is not None:
                _validate_absolute_path(f"mask_paths[{index}]", mask_path)
        if not isinstance(self.anchor_position, int) or isinstance(self.anchor_position, bool):
            raise TypeError("anchor_position must be an integer")
        if not 0 <= self.anchor_position < frame_count:
            raise ValueError("anchor_position must index the sampled frames")
        if self.media_type == "image":
            if self.frame_indices != (0,) or self.anchor_position != 0:
                raise ValueError("image records require frame_indices=(0,) and anchor_position=0")
        elif frame_count < 2:
            raise ValueError("video records require at least two sampled frames")

        if not isinstance(self.query, str) or self.query != self.query.strip():
            raise ValueError("query must be a trimmed string")
        if self.control_kind == "positive":
            if not self.query:
                raise ValueError("positive records require a non-empty query")
            _validate_identifier("target_id", self.target_id)
            if not any(self.target_presence):
                raise ValueError("positive records require the target in at least one frame")
            if not self.target_presence[self.anchor_position]:
                raise ValueError("positive record anchor must contain the target")
            for mask_path, present in zip(self.mask_paths, self.target_presence):
                if present != (mask_path is not None):
                    raise ValueError("positive mask paths must exist exactly on present frames")
        else:
            if self.target_id is not None:
                raise ValueError("negative controls must not define target_id")
            if any(self.target_presence) or any(path is not None for path in self.mask_paths):
                raise ValueError("negative controls must not contain target masks or presence")
            if self.control_kind == "no_object" and not self.query:
                raise ValueError("no_object controls require a non-empty query")
            if self.control_kind == "empty_query" and self.query:
                raise ValueError("empty_query controls require query='' exactly")

    @property
    def optimization_eligible(self) -> bool:
        """Whether the record may enter a future gradient-bearing batch."""

        return self.control_kind != "empty_query"

    @property
    def media_signature(self) -> tuple[Any, ...]:
        """Fields that must stay identical for all queries over one source media."""

        return (
            self.media_type,
            self.media_path,
            self.frame_indices,
            self.anchor_position,
            self.split,
        )


@dataclass(frozen=True)
class QuerySwapPair:
    """A symmetric same-media, different-target query-swap declaration."""

    left_sample_id: str
    right_sample_id: str

    def __post_init__(self) -> None:
        left = _validate_identifier("left_sample_id", self.left_sample_id)
        right = _validate_identifier("right_sample_id", self.right_sample_id)
        if left == right:
            raise ValueError("query-swap samples must be distinct")
        if right < left:
            object.__setattr__(self, "left_sample_id", right)
            object.__setattr__(self, "right_sample_id", left)


@dataclass(frozen=True)
class MaskTrainingManifest:
    """Frozen S4a records, negative coverage, split isolation, and query swaps."""

    records: Tuple[MaskTrainingRecord, ...]
    query_swap_pairs: Tuple[QuerySwapPair, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple) or not self.records:
            raise TypeError("records must be a non-empty tuple")
        if not all(isinstance(record, MaskTrainingRecord) for record in self.records):
            raise TypeError("records must contain MaskTrainingRecord values")
        if not isinstance(self.query_swap_pairs, tuple):
            raise TypeError("query_swap_pairs must be a tuple")
        if not all(isinstance(pair, QuerySwapPair) for pair in self.query_swap_pairs):
            raise TypeError("query_swap_pairs must contain QuerySwapPair values")

        records_by_id = {}
        media_signatures = {}
        media_path_splits = {}
        for record in self.records:
            if record.sample_id in records_by_id:
                raise ValueError(f"duplicate sample_id: {record.sample_id}")
            records_by_id[record.sample_id] = record
            previous = media_signatures.setdefault(record.media_id, record.media_signature)
            if previous != record.media_signature:
                raise ValueError(
                    f"media_id {record.media_id} has inconsistent source frames, anchor, or split"
                )
            previous_split = media_path_splits.setdefault(record.media_path, record.split)
            if previous_split != record.split:
                raise ValueError("the same media_path must not appear in multiple splits")

        positive_media = {
            (record.media_id, record.split)
            for record in self.records
            if record.control_kind == "positive"
        }
        for record in self.records:
            if record.control_kind != "positive" and (record.media_id, record.split) not in positive_media:
                raise ValueError(
                    f"negative control {record.sample_id} must share media with a positive sample"
                )

        seen_pairs = set()
        pair_splits = set()
        for pair in self.query_swap_pairs:
            pair_key = (pair.left_sample_id, pair.right_sample_id)
            if pair_key in seen_pairs:
                raise ValueError(f"duplicate query-swap pair: {pair_key}")
            seen_pairs.add(pair_key)
            try:
                left = records_by_id[pair.left_sample_id]
                right = records_by_id[pair.right_sample_id]
            except KeyError as error:
                raise ValueError(f"query-swap pair references unknown sample: {error.args[0]}") from error
            if left.control_kind != "positive" or right.control_kind != "positive":
                raise ValueError("query-swap pairs require two positive records")
            if left.media_id != right.media_id or left.media_signature != right.media_signature:
                raise ValueError("query-swap pairs must use identical source media, frames, anchor, and split")
            if left.query == right.query:
                raise ValueError("query-swap pairs require different query text")
            if left.target_id == right.target_id:
                raise ValueError("query-swap pairs require different target IDs")
            pair_splits.add(left.split)

        represented_control_splits = {record.split for record in self.records if record.split in ("train", "val")}
        for split in represented_control_splits:
            kinds = {record.control_kind for record in self.records if record.split == split}
            missing = {"no_object", "empty_query"}.difference(kinds)
            if missing:
                raise ValueError(f"split {split} is missing negative controls: {sorted(missing)}")
            if split not in pair_splits:
                raise ValueError(f"split {split} requires a same-media different-target query-swap pair")

    @property
    def summary(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "record_count": len(self.records),
                "query_swap_pair_count": len(self.query_swap_pairs),
                "by_split": _counter_mapping(record.split for record in self.records),
                "by_media_type": _counter_mapping(record.media_type for record in self.records),
                "by_control_kind": _counter_mapping(record.control_kind for record in self.records),
                "optimization_eligible_count": sum(record.optimization_eligible for record in self.records),
            }
        )


@dataclass(frozen=True)
class RetentionProbe:
    """One ordinary-VILA input and its external frozen-baseline output location."""

    probe_id: str
    task: RetentionTask
    prompt: str
    media_paths: Tuple[str, ...]
    comparison: RetentionComparison
    baseline_output_path: str

    def __post_init__(self) -> None:
        _validate_identifier("probe_id", self.probe_id)
        if self.task not in ("text", "single_image", "multi_image", "video"):
            raise ValueError("invalid retention task")
        if not isinstance(self.prompt, str) or not self.prompt or self.prompt != self.prompt.strip():
            raise ValueError("retention prompt must be a non-empty, trimmed string")
        if not isinstance(self.media_paths, tuple):
            raise TypeError("media_paths must be a tuple")
        for index, media_path in enumerate(self.media_paths):
            _validate_absolute_path(f"media_paths[{index}]", media_path)
        expected = {
            "text": len(self.media_paths) == 0,
            "single_image": len(self.media_paths) == 1,
            "multi_image": len(self.media_paths) >= 2,
            "video": len(self.media_paths) == 1,
        }[self.task]
        if not expected:
            raise ValueError(f"media_paths cardinality does not match retention task {self.task}")
        if self.comparison not in ("exact_logits", "exact_tokens", "task_metric"):
            raise ValueError("invalid retention comparison mode")
        _validate_absolute_path("baseline_output_path", self.baseline_output_path)


@dataclass(frozen=True)
class RetentionProbeSet:
    """Frozen ordinary-VILA baseline identity and complete retention probe suite."""

    baseline_model_id: str
    baseline_revision: str
    probes: Tuple[RetentionProbe, ...]

    def __post_init__(self) -> None:
        _validate_identifier("baseline_model_id", self.baseline_model_id)
        _validate_identifier("baseline_revision", self.baseline_revision)
        if not isinstance(self.probes, tuple) or not self.probes:
            raise TypeError("probes must be a non-empty tuple")
        if not all(isinstance(probe, RetentionProbe) for probe in self.probes):
            raise TypeError("probes must contain RetentionProbe values")
        probe_ids = [probe.probe_id for probe in self.probes]
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError("retention probe IDs must be unique")
        output_paths = [probe.baseline_output_path for probe in self.probes]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("retention baseline output paths must be unique")
        tasks = {probe.task for probe in self.probes}
        required = {"text", "single_image", "multi_image", "video"}
        if tasks != required:
            raise ValueError(f"retention probes must cover all tasks; missing {sorted(required - tasks)}")

    @property
    def summary(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "baseline_model_id": self.baseline_model_id,
                "baseline_revision": self.baseline_revision,
                "probe_count": len(self.probes),
                "by_task": _counter_mapping(probe.task for probe in self.probes),
                "by_comparison": _counter_mapping(probe.comparison for probe in self.probes),
                "execution_path": "ordinary_vila",
                "segmentation_enabled": False,
            }
        )
