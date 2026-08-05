import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from llava.evo_seg.training_contracts import (
    MaskTrainingManifest,
    MaskTrainingRecord,
    QuerySwapPair,
    RetentionProbe,
    RetentionProbeSet,
)


def _image_record(sample_id, target_id, query, *, kind="positive", split="train", media_id="image-1"):
    present = kind == "positive"
    return MaskTrainingRecord(
        sample_id=sample_id,
        media_id=media_id,
        media_type="image",
        media_path="/external/images/source.jpg",
        frame_indices=(0,),
        mask_paths=(f"/external/masks/{sample_id}.png" if present else None,),
        target_presence=(present,),
        anchor_position=0,
        split=split,
        query=query,
        target_id=target_id if present else None,
        control_kind=kind,
    )


def _valid_train_manifest():
    records = (
        _image_record("cat", "cat-id", "the cat"),
        _image_record("dog", "dog-id", "the dog"),
        _image_record("absent", None, "the bicycle", kind="no_object"),
        _image_record("empty", None, "", kind="empty_query"),
    )
    return MaskTrainingManifest(records, (QuerySwapPair("dog", "cat"),))


def _retention_probe(probe_id, task, media_paths):
    return RetentionProbe(
        probe_id=probe_id,
        task=task,
        prompt=f"ordinary {task} prompt",
        media_paths=media_paths,
        comparison="exact_logits",
        baseline_output_path=f"/external/retention/{probe_id}.json",
    )


def _retention_probes():
    return (
        _retention_probe("text", "text", ()),
        _retention_probe("single", "single_image", ("/external/images/a.jpg",)),
        _retention_probe(
            "multi",
            "multi_image",
            ("/external/images/a.jpg", "/external/images/b.jpg"),
        ),
        _retention_probe("video", "video", ("/external/videos/a.mp4",)),
    )


def test_image_records_distinguish_optimization_and_fail_closed_controls():
    records = _valid_train_manifest().records
    positive, no_object, empty = records[0], records[2], records[3]
    assert positive.optimization_eligible
    assert no_object.optimization_eligible
    assert not empty.optimization_eligible
    assert empty.query == ""
    with pytest.raises(FrozenInstanceError):
        positive.query = "changed"


def test_video_record_freezes_source_frame_mask_and_anchor_alignment():
    record = MaskTrainingRecord(
        sample_id="video-target",
        media_id="video-1",
        media_type="video",
        media_path="/external/videos/source.mp4",
        frame_indices=(4, 8, 12),
        mask_paths=(None, "/external/masks/00008.png", "/external/masks/00012.png"),
        target_presence=(False, True, True),
        anchor_position=1,
        split="val",
        query="the red vehicle",
        target_id="vehicle-3",
    )
    assert record.frame_indices[record.anchor_position] == 8
    assert record.media_signature == (
        "video",
        "/external/videos/source.mp4",
        (4, 8, 12),
        1,
        "val",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"media_path": "relative.jpg"}, "absolute external"),
        ({"frame_indices": (1, 1)}, "strictly increasing"),
        ({"mask_paths": ("/external/mask.png", None)}, "aligned"),
        ({"target_presence": (False,)}, "at least one frame"),
        ({"anchor_position": 1}, "sampled frames"),
        ({"query": ""}, "non-empty query"),
    ],
)
def test_positive_record_rejects_ambiguous_metadata(changes, message):
    values = dict(
        sample_id="sample",
        media_id="image",
        media_type="image",
        media_path="/external/image.jpg",
        frame_indices=(0,),
        mask_paths=("/external/mask.png",),
        target_presence=(True,),
        anchor_position=0,
        split="train",
        query="the object",
        target_id="object",
        control_kind="positive",
    )
    values.update(changes)
    with pytest.raises((TypeError, ValueError), match=message):
        MaskTrainingRecord(**values)


def test_negative_controls_cannot_carry_targets_or_mismatched_queries():
    with pytest.raises(ValueError, match="must not contain target"):
        MaskTrainingRecord(
            sample_id="bad-negative",
            media_id="image",
            media_type="image",
            media_path="/external/image.jpg",
            frame_indices=(0,),
            mask_paths=("/external/mask.png",),
            target_presence=(True,),
            anchor_position=0,
            split="train",
            query="absent object",
            target_id=None,
            control_kind="no_object",
        )
    with pytest.raises(ValueError, match="query='' exactly"):
        _image_record("bad-empty", None, "not empty", kind="empty_query")


def test_manifest_validates_query_swaps_negative_coverage_and_summary():
    manifest = _valid_train_manifest()
    pair = manifest.query_swap_pairs[0]
    assert (pair.left_sample_id, pair.right_sample_id) == ("cat", "dog")
    assert manifest.summary["record_count"] == 4
    assert manifest.summary["optimization_eligible_count"] == 3
    assert manifest.summary["by_control_kind"] == {
        "empty_query": 1,
        "no_object": 1,
        "positive": 2,
    }
    with pytest.raises(TypeError):
        manifest.summary["record_count"] = 0


def test_manifest_rejects_missing_train_or_validation_controls():
    records = (
        _image_record("cat", "cat-id", "the cat"),
        _image_record("dog", "dog-id", "the dog"),
    )
    with pytest.raises(ValueError, match="missing negative controls"):
        MaskTrainingManifest(records, (QuerySwapPair("cat", "dog"),))


def test_manifest_rejects_source_media_split_leakage():
    records = _valid_train_manifest().records + (
        _image_record("val-cat", "cat-id", "the cat", split="val"),
    )
    with pytest.raises(ValueError, match="inconsistent source frames, anchor, or split"):
        MaskTrainingManifest(records, (QuerySwapPair("cat", "dog"),))

    records_with_relabelled_media = _valid_train_manifest().records + (
        _image_record(
            "val-cat",
            "cat-id",
            "the cat",
            split="val",
            media_id="relabeled-image",
        ),
    )
    with pytest.raises(ValueError, match="media_path must not appear in multiple splits"):
        MaskTrainingManifest(records_with_relabelled_media, (QuerySwapPair("cat", "dog"),))


def test_manifest_rejects_cross_media_or_same_target_query_swaps():
    controls = (
        _image_record("absent", None, "the bicycle", kind="no_object"),
        _image_record("empty", None, "", kind="empty_query"),
    )
    cross_media = (
        _image_record("cat", "cat-id", "the cat"),
        _image_record("dog", "dog-id", "the dog", media_id="image-2"),
    ) + controls
    with pytest.raises(ValueError, match="identical source media"):
        MaskTrainingManifest(cross_media, (QuerySwapPair("cat", "dog"),))
    same_target = (
        _image_record("cat-a", "cat-id", "the left cat"),
        _image_record("cat-b", "cat-id", "the right cat"),
    ) + controls
    with pytest.raises(ValueError, match="different target IDs"):
        MaskTrainingManifest(same_target, (QuerySwapPair("cat-a", "cat-b"),))
    same_query = (
        _image_record("cat", "cat-id", "the animal"),
        _image_record("dog", "dog-id", "the animal"),
    ) + controls
    with pytest.raises(ValueError, match="different query text"):
        MaskTrainingManifest(same_query, (QuerySwapPair("cat", "dog"),))


def test_retention_probe_set_requires_all_ordinary_vila_tasks():
    probes = _retention_probes()
    probe_set = RetentionProbeSet(
        baseline_model_id="VILA1.5-3b",
        baseline_revision="0f1426e8da9181e6e6653e10bc15f62d515fa2f6",
        probes=probes,
    )
    assert probe_set.summary["execution_path"] == "ordinary_vila"
    assert probe_set.summary["segmentation_enabled"] is False
    assert probe_set.summary["by_task"] == {
        "multi_image": 1,
        "single_image": 1,
        "text": 1,
        "video": 1,
    }
    with pytest.raises(ValueError, match="cover all tasks"):
        RetentionProbeSet("VILA1.5-3b", "revision", probes[:-1])


def test_retention_probe_rejects_wrong_media_cardinality_and_relative_outputs():
    with pytest.raises(ValueError, match="cardinality"):
        _retention_probe("bad-text", "text", ("/external/image.jpg",))
    with pytest.raises(ValueError, match="absolute external"):
        RetentionProbe(
            probe_id="bad-output",
            task="video",
            prompt="Summarize the video.",
            media_paths=("/external/video.mp4",),
            comparison="exact_tokens",
            baseline_output_path="retention.json",
        )


def test_importing_training_contracts_does_not_import_vila_models_or_sam2():
    command = (
        "import sys; import llava.evo_seg.training_contracts; "
        "assert 'llava.model' not in sys.modules; "
        "assert not any(name == 'sam2' or name.startswith('sam2.') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
