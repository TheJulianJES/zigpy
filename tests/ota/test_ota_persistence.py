"""Tests for persisting OTA provider index metadata."""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import typing
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from tests.ota.test_ota_providers import make_device
from zigpy import config
import zigpy.ota
from zigpy.ota.image import FieldControl
from zigpy.ota.providers import (
    BaseOtaImageMetadata,
    BaseOtaProvider,
    LocalOtaImageMetadata,
    RemoteOtaImageMetadata,
    deserialize_image_metadata,
    serialize_image_metadata,
)
from zigpy.zcl.clusters.general import Ota


@pytest.fixture
def query_cmd():
    """Common query command for OTA tests."""
    return Ota.ServerCommandDefs.query_next_image.schema(
        field_control=FieldControl.HARDWARE_VERSIONS_PRESENT,
        manufacturer_code=0x1234,
        image_type=0xABCD,
        current_file_version=1,
        hardware_version=1,
    )


@pytest.fixture
def remote_meta():
    """Persistable trusted image metadata compatible with `query_cmd`."""
    return RemoteOtaImageMetadata(
        file_version=2,
        manufacturer_id=0x1234,
        image_type=0xABCD,
        checksum="sha3-256:" + hashlib.sha3_256(b"firmware").hexdigest(),
        file_size=8,
        url="https://example.org/firmware.ota",
    )


class TrustedRemoteProvider(BaseOtaProvider):
    """Trusted provider serving persistable metadata."""

    TRUSTED = True

    def __init__(self, index: list[BaseOtaImageMetadata]) -> None:
        super().__init__()
        self._index = index

    def compatible_with_device(self, device) -> bool:
        return True

    async def _load_index(
        self, session: aiohttp.ClientSession
    ) -> typing.AsyncIterator[BaseOtaImageMetadata]:
        for meta in self._index:
            yield meta


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


async def test_trusted_provider_index_is_persisted(query_cmd, remote_meta) -> None:
    """A trusted provider's refreshed index is sent to application listeners."""
    device = make_device(model="device model", manufacturer_id=0x1234)

    app = MagicMock()
    ota = zigpy.ota.OTA(config={config.CONF_OTA_ENABLED: False}, application=app)
    provider = TrustedRemoteProvider([remote_meta])
    ota.register_provider(provider)

    await ota.get_ota_images(device, query_cmd)

    app.listener_event.assert_called_once_with(
        "ota_provider_index_updated",
        repr(provider),
        provider._index_last_updated,
        [serialize_image_metadata(remote_meta.replace(trusted=True))],
    )

    # A cached (not refreshed) index is not re-persisted
    app.listener_event.reset_mock()
    await ota.get_ota_images(device, query_cmd)
    app.listener_event.assert_not_called()


async def test_untrusted_provider_index_is_not_persisted(query_cmd) -> None:
    """Untrusted provider indexes are not persisted."""
    device = make_device(model="device model", manufacturer_id=0x1234)

    class UntrustedRemoteProvider(TrustedRemoteProvider):
        TRUSTED = False

    app = MagicMock()
    ota = zigpy.ota.OTA(config={config.CONF_OTA_ENABLED: False}, application=app)
    ota.register_provider(UntrustedRemoteProvider([]))

    await ota.get_ota_images(device, query_cmd)
    app.listener_event.assert_not_called()


async def test_restore_cached_index(query_cmd, remote_meta) -> None:
    """A persisted index is restored and served without network access."""
    device = make_device(model="device model", manufacturer_id=0x1234)

    ota = zigpy.ota.OTA(config={config.CONF_OTA_ENABLED: False}, application=None)
    provider = TrustedRemoteProvider([])
    ota.register_provider(provider)

    last_updated = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)

    # Unknown provider identities and untrusted providers do not match
    assert not ota.restore_cached_index("unknown", last_updated, [remote_meta])

    assert ota.restore_cached_index(repr(provider), last_updated, [remote_meta])

    # The image is served from the restored cache, without a network refresh
    with patch.object(provider, "_load_index", wraps=provider._load_index) as load:
        images = await ota.get_ota_images(device, query_cmd)

    assert len(load.mock_calls) == 0
    assert len(images.upgrades) == 1
    assert images.upgrades[0].metadata == remote_meta.replace(trusted=True)

    # The restored freshness is capped so the index expires shortly after boot
    now = datetime.datetime.now(datetime.UTC)
    expires_at = provider._index_last_updated + provider.INDEX_EXPIRATION_TIME
    assert expires_at <= now + zigpy.ota.POST_RESTORE_REFRESH_DELAY_MAX

    # A post-restore refresh was scheduled
    assert ota._post_restore_refresh_task is not None
    ota.stop_periodic_broadcasts()
    assert ota._post_restore_refresh_task is None


async def test_restore_cached_index_expired(query_cmd, remote_meta) -> None:
    """An expired persisted index is refreshed on the first device check."""
    device = make_device(model="device model", manufacturer_id=0x1234)

    ota = zigpy.ota.OTA(config={config.CONF_OTA_ENABLED: False}, application=None)
    provider = TrustedRemoteProvider([remote_meta.replace(file_version=3)])
    ota.register_provider(provider)

    last_updated = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=2)
    assert ota.restore_cached_index(repr(provider), last_updated, [remote_meta])

    # The first check refreshes immediately and serves the live index
    with patch.object(provider, "_load_index", wraps=provider._load_index) as load:
        images = await ota.get_ota_images(device, query_cmd)

    assert len(load.mock_calls) == 1
    assert len(images.upgrades) == 1
    assert images.upgrades[0].metadata.file_version == 3

    ota.stop_periodic_broadcasts()


async def test_restore_cached_index_does_not_overwrite_live_index(
    query_cmd, remote_meta
) -> None:
    """A live index is not overwritten by a stale restore."""
    device = make_device(model="device model", manufacturer_id=0x1234)

    ota = zigpy.ota.OTA(config={config.CONF_OTA_ENABLED: False}, application=None)
    provider = TrustedRemoteProvider([remote_meta])
    ota.register_provider(provider)

    images1 = await ota.get_ota_images(device, query_cmd)
    assert len(images1.upgrades) == 1

    # Restoring after a live refresh is a no-op
    last_updated = datetime.datetime.now(datetime.UTC)
    assert ota.restore_cached_index(repr(provider), last_updated, [])

    images2 = await ota.get_ota_images(device, query_cmd)
    assert images2 == images1
    assert ota._post_restore_refresh_task is None


async def test_post_restore_refresh_checks_devices() -> None:
    """The post-restore refresh task re-checks all devices after the delay."""
    ota = zigpy.ota.OTA(config={config.CONF_OTA_ENABLED: False}, application=None)
    ota.check_all_devices_for_ota = AsyncMock()

    with patch("asyncio.sleep") as sleep:
        await ota._post_restore_refresh(123.0)

    sleep.assert_called_once_with(123.0)
    ota.check_all_devices_for_ota.assert_called_once_with()
