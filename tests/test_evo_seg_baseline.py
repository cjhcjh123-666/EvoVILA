from types import SimpleNamespace

from llava import Image, Video
from llava.evo_seg import SegmentationCapability
from llava.utils.media import extract_media


def extract_draft(value, num_video_frames=4):
    conversation = [{"from": "human", "value": value}]
    config = SimpleNamespace(num_video_frames=num_video_frames)
    media = extract_media(conversation, config=config, draft=True)
    return conversation[0]["value"], media


def test_original_text_only_prompt_contract_with_extension_disabled():
    text, media = extract_draft("Describe the scene.")
    assert text == "Describe the scene."
    assert dict(media) == {}
    assert not SegmentationCapability.is_enabled(None)


def test_original_single_image_prompt_contract_with_extension_disabled():
    image = Image("/external/not-opened/image.jpg")
    text, media = extract_draft([image, "What is visible?"])
    assert text == "<image>What is visible?"
    assert media["image"] == [image]
    assert not SegmentationCapability.is_enabled(None)


def test_original_multi_image_prompt_contract_with_extension_disabled():
    images = [Image("/external/not-opened/first.jpg"), Image("/external/not-opened/second.jpg")]
    text, media = extract_draft([images[0], "Compare them. ", images[1]])
    assert text == "<image>Compare them. <image>"
    assert media["image"] == images
    assert not SegmentationCapability.is_enabled(None)


def test_original_video_prompt_contract_with_extension_disabled():
    video = Video("/external/not-opened/video.mp4")
    text, media = extract_draft([video, "Summarize the video."], num_video_frames=3)
    assert text == "<image><image><image>Summarize the video."
    assert media["image"] == [video]
    assert not SegmentationCapability.is_enabled(None)
