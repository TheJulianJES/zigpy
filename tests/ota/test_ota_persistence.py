"""Tests for persisting OTA provider index metadata."""

from __future__ import annotations

import json
import pathlib

from zigpy.ota.providers import (
    LocalOtaImageMetadata,
    RemoteOtaImageMetadata,
    deserialize_image_metadata,
    serialize_image_metadata,
)


def test_metadata_serialization_roundtrip_remote() -> None:
    meta = RemoteOtaImageMetadata(
        file_version=0x12345678,
        manufacturer_id=4476,
        image_type=0x2101,
        checksum="sha3-256:" + "ab" * 32,
        file_size=12345,
        manufacturer_names=("IKEA of Sweden",),
        model_names=("TRADFRI bulb",),
        changelog="Changelog",
        release_notes="Notes",
        min_current_file_version=1,
        max_current_file_version=0x12345677,
        specificity=2,
        source="zigpy-ota",
        trusted=True,
        url="https://example.org/firmware.ota",
    )

    obj = serialize_image_metadata(meta)
    assert obj is not None

    # The serialized form is JSON-compatible
    restored = deserialize_image_metadata(json.loads(json.dumps(obj)))
    assert restored == meta
    assert hash(restored) == hash(meta)


def test_metadata_serialization_roundtrip_local(tmp_path: pathlib.Path) -> None:
    meta = LocalOtaImageMetadata(
        file_version=2,
        path=tmp_path / "firmware.ota",
        trusted=True,
    )

    obj = serialize_image_metadata(meta)
    restored = deserialize_image_metadata(json.loads(json.dumps(obj)))
    assert restored == meta
    assert isinstance(restored.path, pathlib.Path)


def test_metadata_serialization_unknown_type() -> None:
    class CustomMetadata(RemoteOtaImageMetadata):
        pass

    # Unknown subclasses are not serialized
    assert (
        serialize_image_metadata(
            CustomMetadata(file_version=1, url="https://example.org/fw.ota")
        )
        is None
    )


def test_metadata_deserialization_is_defensive() -> None:
    # Not a dict
    assert deserialize_image_metadata(None) is None
    assert deserialize_image_metadata([1, 2, 3]) is None

    # Missing or invalid type tag
    assert deserialize_image_metadata({"file_version": 1}) is None
    assert deserialize_image_metadata({"type": ["remote"], "file_version": 1}) is None
    assert deserialize_image_metadata({"type": "from_the_future"}) is None

    # Missing required fields
    assert deserialize_image_metadata({"type": "remote", "file_version": 1}) is None
    assert (
        deserialize_image_metadata({"type": "remote", "url": "https://example.org"})
        is None
    )

    # Unknown fields written by a future zigpy version are dropped
    restored = deserialize_image_metadata(
        {
            "type": "remote",
            "file_version": 1,
            "url": "https://example.org/fw.ota",
            "field_from_the_future": "ignored",
        }
    )
    assert restored == RemoteOtaImageMetadata(
        file_version=1, url="https://example.org/fw.ota"
    )
