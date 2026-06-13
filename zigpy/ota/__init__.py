"""OTA support for Zigbee devices."""

from __future__ import annotations

import asyncio
from asyncio import timeout as asyncio_timeout
from collections import defaultdict
import contextlib
import dataclasses
import datetime
import hashlib
import logging
import random
import typing

from zigpy.config import (
    CONF_OTA_ADVANCED_DIR,
    CONF_OTA_ALLOW_ADVANCED_DIR,
    CONF_OTA_DISABLE_DEFAULT_PROVIDERS,
    CONF_OTA_ENABLED,
    CONF_OTA_EXTRA_PROVIDERS,
    CONF_OTA_IKEA,
    CONF_OTA_LEDVANCE,
    CONF_OTA_PROVIDER_MANUF_IDS,
    CONF_OTA_PROVIDER_URL,
    CONF_OTA_PROVIDERS,
    CONF_OTA_REMOTE_PROVIDERS,
    CONF_OTA_SALUS,
    CONF_OTA_SONOFF,
    CONF_OTA_Z2M_LOCAL_INDEX,
    CONF_OTA_Z2M_REMOTE_INDEX,
)
from zigpy.ota.image import BaseOTAImage
import zigpy.ota.providers
import zigpy.profiles.zha
import zigpy.types as t
import zigpy.util
from zigpy.zcl import OtaImageAvailableEvent, foundation
from zigpy.zcl.clusters.general import Ota, QueryNextImageCommand

if typing.TYPE_CHECKING:
    import zigpy.application
    import zigpy.device

_LOGGER = logging.getLogger(__name__)

OTA_FETCH_TIMEOUT = 20
MAX_DEVICES_CHECKING_IN_PER_BROADCAST = 15
BROADCAST_SETTLE_DELAY = 60

# How long after startup indexes restored from the database are refreshed.
# Users expect a restart to eventually pick up new firmware, but refreshing
# immediately would defeat the index rate limiting on every restart. The delay
# is randomized per provider so a fleet of instances restarting simultaneously
# (e.g. after a Home Assistant update) does not hit the index servers at once.
POST_RESTORE_REFRESH_DELAY_MIN = datetime.timedelta(minutes=10)
POST_RESTORE_REFRESH_DELAY_MAX = datetime.timedelta(minutes=30)

# Size limit for the downloaded firmware cache. Firmware is re-downloaded on
# demand if it is evicted, so the limit only trades memory for bandwidth.
FIRMWARE_CACHE_SIZE_LIMIT = 32 * 1024 * 1024  # bytes


@dataclasses.dataclass(frozen=True)
class OtaImagesResult(t.BaseDataclassMixin):
    upgrades: tuple[OtaImageWithMetadata, ...]
    downgrades: tuple[OtaImageWithMetadata, ...]


@dataclasses.dataclass(frozen=True)
class OtaImageWithMetadata(t.BaseDataclassMixin):
    metadata: zigpy.ota.providers.BaseOtaImageMetadata
    firmware: BaseOTAImage | None

    def __repr__(self) -> str:
        if self.firmware is not None:
            firmware_repr = (
                f"<{type(self.firmware).__name__}: "
                f"{self.firmware.header.image_size} bytes>"
            )
        else:
            firmware_repr = "None"

        return (
            f"{type(self).__name__}("
            f"metadata={self.metadata!r}, "
            f"firmware={firmware_repr})"
        )

    @property
    def version(self) -> int:
        return self.metadata.file_version

    @property
    def _min_hardware_version(self) -> int | None:
        if self.metadata.min_hardware_version is not None:
            return self.metadata.min_hardware_version
        elif (
            self.firmware is not None
            and self.firmware.header.minimum_hardware_version is not None
        ):
            return self.firmware.header.minimum_hardware_version
        else:
            return None

    @property
    def _max_hardware_version(self) -> int | None:
        if self.metadata.max_hardware_version is not None:
            return self.metadata.max_hardware_version
        elif (
            self.firmware is not None
            and self.firmware.header.maximum_hardware_version is not None
        ):
            return self.firmware.header.maximum_hardware_version
        else:
            return None

    @property
    def _manufacturer_id(self) -> int | None:
        if self.metadata.manufacturer_id is not None:
            return self.metadata.manufacturer_id
        elif self.firmware is not None:
            return self.firmware.header.manufacturer_id
        else:
            return None

    @property
    def _image_type(self) -> int | None:
        if self.metadata.image_type is not None:
            return self.metadata.image_type
        elif self.firmware is not None:
            return self.firmware.header.image_type
        else:
            return None

    @property
    def specificity(self) -> int:
        """Return a numerical representation of the metadata specificity.
        Higher specificity is preferred to lower when picking a final OTA image.
        """

        total = 0

        if self.metadata.manufacturer_names:
            total += 1000

        if self.metadata.model_names:
            total += 1000

        if self._image_type is not None:
            total += 100

        if self._manufacturer_id is not None:
            total += 100

        if self.metadata.min_current_file_version is not None:
            total += 10

        if self.metadata.max_current_file_version is not None:
            total += 10

        if self._min_hardware_version is not None:
            total += 1

        if self._max_hardware_version is not None:
            total += 1

        # Boost the specificity
        if self.metadata.specificity is not None:
            total += self.metadata.specificity

        # Prefer images from trusted providers (e.g. zigpy-ota has richer metadata)
        if self.metadata.trusted:
            total += 10000

        return total

    def check_compatibility(
        self,
        device: zigpy.device.Device,
        query_cmd: QueryNextImageCommand,
    ) -> bool:
        """Check if an OTA image and its metadata is compatible with a device."""
        if (
            self._manufacturer_id is not None
            and self._manufacturer_id != query_cmd.manufacturer_code
        ):
            return False

        if self._image_type is not None and self._image_type != query_cmd.image_type:
            return False

        if self.metadata.model_names and device.model not in self.metadata.model_names:
            return False

        if (
            self.metadata.manufacturer_names
            and device.manufacturer not in self.metadata.manufacturer_names
        ):
            return False

        if self._min_hardware_version is not None and (
            query_cmd.hardware_version is None
            or query_cmd.hardware_version < self._min_hardware_version
        ):
            return False

        if self._max_hardware_version is not None and (
            query_cmd.hardware_version is None
            or query_cmd.hardware_version > self._max_hardware_version
        ):
            return False

        return True

    def check_version(self, current_file_version: int) -> bool:
        """Check if the image is a newer version than the device's current version."""
        if self.version <= current_file_version:
            return False

        if (
            self.metadata.min_current_file_version is not None
            and current_file_version < self.metadata.min_current_file_version
        ):
            return False

        if (
            self.metadata.max_current_file_version is not None
            and current_file_version > self.metadata.max_current_file_version
        ):
            return False

        return True

    async def fetch(self) -> OtaImageWithMetadata:
        firmware = await self.metadata.fetch()

        return self.replace(
            metadata=self.metadata,
            firmware=firmware,
        )


class OTA:
    """OTA Manager."""

    def __init__(
        self,
        config: dict[str, typing.Any],
        application: zigpy.application.ControllerApplication,
    ) -> None:
        self._config = config
        self._application = application

        self._providers: list[zigpy.ota.providers.BaseOtaProvider] = []
        # Per-provider index metadata, replaced wholesale on every refresh
        self._image_cache: dict[
            zigpy.ota.providers.BaseOtaProvider,
            set[zigpy.ota.providers.BaseOtaImageMetadata],
        ] = {}
        # Downloaded firmware, keyed by fetch identity. This is a pure cache:
        # images are only ever served through the index metadata above, so an
        # unreferenced blob is unreachable and any eviction is safe.
        self._firmware_cache: dict[typing.Hashable, BaseOTAImage] = {}

        self._broadcast_loop_task = None
        self._post_restore_refresh_task: asyncio.Task | None = None

        if config[CONF_OTA_ENABLED]:
            self._register_providers(self._config)

    async def broadcast_loop(self, initial_delay: float, interval: float) -> None:
        """Periodically broadcast an image notification to get devices to check in."""

        await asyncio.sleep(initial_delay)

        while True:
            _LOGGER.debug("Broadcasting OTA notification")

            try:
                await self.broadcast_notify()
            except Exception:  # noqa: BLE001
                _LOGGER.debug("OTA broadcast failed", exc_info=True)

            # Wait for devices to respond with query_next_image before checking
            await asyncio.sleep(BROADCAST_SETTLE_DELAY)

            try:
                await self.check_all_devices_for_ota()
            except Exception:  # noqa: BLE001
                _LOGGER.debug("OTA image check failed", exc_info=True)

            await asyncio.sleep(interval)

    def start_periodic_broadcasts(self, initial_delay: float, interval: float) -> None:
        """Start the periodic OTA broadcasts."""
        self._broadcast_loop_task = asyncio.create_task(
            self.broadcast_loop(
                initial_delay=initial_delay,
                interval=interval,
            )
        )

    def stop_periodic_broadcasts(self) -> None:
        """Stop the periodic OTA broadcasts."""
        if self._broadcast_loop_task is not None:
            self._broadcast_loop_task.cancel()
            self._broadcast_loop_task = None

        if self._post_restore_refresh_task is not None:
            self._post_restore_refresh_task.cancel()
            self._post_restore_refresh_task = None

    def invalidate_provider_caches(self) -> None:
        """Invalidate all provider index caches, forcing a refresh on next check.

        The refresh revokes images withdrawn from the new indexes, dropping
        their downloaded firmware. Firmware of images still being served is
        kept, even if cosmetic metadata (e.g. release notes) changed.
        """
        for provider in self._providers:
            provider.invalidate_index()

    async def check_cluster_for_ota(self, cluster: Ota) -> None:
        """Check OTA image availability for a single OTA cluster.

        If the cluster has a cached query command, calls get_ota_images and emits
        OtaImageAvailableEvent on the cluster. Intended to be called by consumers
        (e.g. ZHA) during entity setup after registering their event listener.
        """
        cmd = cluster.last_query_cmd
        if cmd is None:
            return

        device = cluster.endpoint.device
        images_result = await self.get_ota_images(device, cmd)

        cluster.emit(
            OtaImageAvailableEvent.event_type,
            OtaImageAvailableEvent(
                device_ieee=str(device.ieee),
                endpoint_id=cluster.endpoint.endpoint_id,
                cluster_type=cluster.cluster_type,
                cluster_id=cluster.cluster_id,
                images_result=images_result,
                query_cmd=cmd,
            ),
        )

    async def check_device_for_ota(
        self,
        device: zigpy.device.Device,
    ) -> None:
        """Check OTA image availability for a single device.

        Iterates the device's endpoints looking for OTA clusters with cached
        query commands and calls check_cluster_for_ota for each.
        """
        for ep_id, ep in device.endpoints.items():
            if ep_id == 0:
                continue

            # Prefer out_clusters (client) since that's where runtime routing
            # places query_next_image when both cluster types exist. If an
            # out_cluster exists, always use it (the in_cluster's cached query
            # would be stale and never updated again).
            for clusters in (ep.out_clusters, ep.in_clusters):
                cluster = clusters.get(Ota.cluster_id)
                if not isinstance(cluster, Ota):
                    continue

                await self.check_cluster_for_ota(cluster)
                break

    async def check_all_devices_for_ota(self) -> None:
        """Check OTA image availability for all devices with cached query commands.

        Called periodically from the broadcast loop and by consumers (e.g. ZHA)
        for user-initiated "check for updates" after invalidate_provider_caches().
        """
        for device in self._application.devices.values():
            try:
                await self.check_device_for_ota(device)
            except Exception:  # noqa: BLE001
                _LOGGER.debug(
                    "Failed to check OTA images for %s",
                    device.ieee,
                    exc_info=True,
                )

    def _register_providers(self, config: dict[str, typing.Any]) -> None:
        # Config gets a little complicated when you mix deprecated config and the new
        # providers config. We treat every option as an "intent" and merge configs in
        # the end.
        with_providers: list[zigpy.ota.providers.BaseOtaProvider] = [
            *config[CONF_OTA_PROVIDERS],
            *config[CONF_OTA_EXTRA_PROVIDERS],
        ]
        without_providers: set[type[zigpy.ota.providers.BaseOtaProvider]] = set(
            config[CONF_OTA_DISABLE_DEFAULT_PROVIDERS]
        ) - {type(p) for p in config[CONF_OTA_EXTRA_PROVIDERS]}

        def register_deprecated_provider(
            enabled: bool | str | None,
            provider: type[zigpy.ota.providers.BaseOtaProvider],
            config: dict[str, typing.Any] | None = None,
        ) -> None:
            if isinstance(enabled, str) and not config:
                config = {"url": enabled}
                enabled = True

            if not config:
                config = {}

            if enabled is True:
                with_providers.append(provider(**config))

                with contextlib.suppress(KeyError):
                    without_providers.remove(provider)
            elif enabled is False:
                without_providers.add(provider)
            else:
                pass

        register_deprecated_provider(
            enabled=config.get(CONF_OTA_IKEA),
            provider=zigpy.ota.providers.Tradfri,
        )
        register_deprecated_provider(
            enabled=config.get(CONF_OTA_LEDVANCE),
            provider=zigpy.ota.providers.Ledvance,
        )
        register_deprecated_provider(
            enabled=config.get(CONF_OTA_SALUS),
            provider=zigpy.ota.providers.Salus,
        )
        register_deprecated_provider(
            enabled=config.get(CONF_OTA_SONOFF),
            provider=zigpy.ota.providers.Sonoff,
        )
        register_deprecated_provider(
            enabled=config.get(CONF_OTA_Z2M_REMOTE_INDEX),
            provider=zigpy.ota.providers.RemoteZ2MProvider,
        )
        register_deprecated_provider(
            enabled=config.get(CONF_OTA_ALLOW_ADVANCED_DIR),
            provider=zigpy.ota.providers.AdvancedFileProvider,
            config={"path": config.get(CONF_OTA_ADVANCED_DIR)},
        )
        register_deprecated_provider(
            enabled=None if config.get(CONF_OTA_Z2M_LOCAL_INDEX) is None else True,
            provider=zigpy.ota.providers.LocalZ2MProvider,
            config={"index_file": config.get(CONF_OTA_Z2M_LOCAL_INDEX)},
        )

        for provider_config in config.get(CONF_OTA_REMOTE_PROVIDERS, []):
            register_deprecated_provider(
                enabled=True,
                provider=zigpy.ota.providers.RemoteZigpyProvider,
                config={
                    "url": provider_config[CONF_OTA_PROVIDER_URL],
                    "manufacturer_ids": provider_config[CONF_OTA_PROVIDER_MANUF_IDS],
                },
            )

        replaced_providers: list[zigpy.ota.providers.BaseOtaProvider] = []

        for provider in with_providers:
            if type(provider) in without_providers:
                continue

            if provider.override_previous:
                replaced_providers = [
                    p for p in replaced_providers if type(p) is not type(provider)
                ]

            replaced_providers.append(provider)

        for provider in replaced_providers:
            self.register_provider(provider)

    def register_provider(self, provider: zigpy.ota.providers.BaseOtaProvider) -> None:
        """Register a new OTA provider."""
        _LOGGER.debug("Registering new OTA provider: %s", provider)
        self._providers.append(provider)

    @zigpy.util.combine_concurrent_calls
    async def _refresh_provider_index(
        self, provider: zigpy.ota.providers.BaseOtaProvider
    ) -> None:
        """Load a provider's index, if it expired, and rebuild its image cache.

        Concurrent calls for the same provider are combined so a burst of device
        checks (e.g. at startup) downloads, caches, and logs each index once.
        """
        try:
            async with asyncio_timeout(OTA_FETCH_TIMEOUT):
                index = await provider.load_index()
        except Exception as exc:  # noqa: BLE001
            # Keep the previously-cached images: a brief provider outage should
            # not withdraw its images
            _LOGGER.debug("Failed to load provider %s", provider, exc_info=exc)
            provider.record_index_failure()
            self._expire_stale_provider_images(provider)
            return

        # The cached index is still fresh
        if index is None:
            return

        _LOGGER.debug("Loaded %d images from provider: %s", len(index), provider)

        new_index: set[zigpy.ota.providers.BaseOtaImageMetadata] = set()

        for meta in index:
            # Mark metadata as trusted if it comes from a trusted provider
            if provider.TRUSTED and not meta.trusted:
                meta = meta.replace(trusted=True)

            # Trusted images are not downloaded before installation, so their
            # content cannot be verified without a SHA3-256 checksum
            if meta.trusted and (
                meta.checksum is None or not meta.checksum.startswith("sha3-256:")
            ):
                _LOGGER.warning(
                    "Trusted image %s does not have SHA3-256 checksum, ignoring",
                    meta,
                )
                continue

            new_index.add(meta)

        # Replace the provider's index wholesale so images withdrawn from the
        # index are revoked, then drop firmware nothing references anymore
        self._image_cache[provider] = new_index
        self._prune_firmware_cache()

        if provider.TRUSTED:
            self._persist_provider_index(provider, new_index)

    def _persist_provider_index(
        self,
        provider: zigpy.ota.providers.BaseOtaProvider,
        metadata: set[zigpy.ota.providers.BaseOtaImageMetadata],
    ) -> None:
        """Notify listeners (i.e. the database) of a trusted provider's new index."""
        if self._application is None:
            return

        index = [
            obj
            for obj in (
                zigpy.ota.providers.serialize_image_metadata(meta) for meta in metadata
            )
            if obj is not None
        ]

        self._application.listener_event(
            "ota_provider_index_updated",
            repr(provider),
            provider._index_last_updated,
            index,
        )

    def restore_cached_index(
        self,
        provider_id: str,
        last_updated: datetime.datetime,
        metadata: list[zigpy.ota.providers.BaseOtaImageMetadata],
    ) -> bool:
        """Restore a trusted provider's index cache from the database.

        Returns whether a registered trusted provider matched the persisted
        identity. Restored indexes are refreshed shortly after startup, or on
        the first device check if the persisted index already expired.
        """
        provider = next(
            (p for p in self._providers if p.TRUSTED and repr(p) == provider_id),
            None,
        )

        if provider is None:
            return False

        if provider in self._image_cache:
            # The provider has already loaded a live index
            return True

        images: set[zigpy.ota.providers.BaseOtaImageMetadata] = set()

        for meta in metadata:
            if not meta.trusted:
                meta = meta.replace(trusted=True)

            images.add(meta)

        self._image_cache[provider] = images

        # Cap the restored freshness so the index expires (and is refreshed)
        # shortly after startup instead of inheriting the full remaining TTL
        now = datetime.datetime.now(datetime.UTC)
        refresh_delay = datetime.timedelta(
            seconds=random.uniform(  # noqa: S311
                POST_RESTORE_REFRESH_DELAY_MIN.total_seconds(),
                POST_RESTORE_REFRESH_DELAY_MAX.total_seconds(),
            )
        )
        provider._index_last_updated = min(
            last_updated,
            now + refresh_delay - provider.INDEX_EXPIRATION_TIME,
        )
        # The restored index was downloaded successfully at `last_updated`, so
        # a failed refresh right after startup must not count it as stale
        provider._index_last_success = last_updated

        # Re-check all devices once every restored index has expired
        if self._post_restore_refresh_task is None:
            self._post_restore_refresh_task = asyncio.create_task(
                self._post_restore_refresh(
                    POST_RESTORE_REFRESH_DELAY_MAX.total_seconds() + 1
                )
            )

        return True

    async def _post_restore_refresh(self, delay: float) -> None:
        """Refresh the restored indexes and re-check all devices."""
        await asyncio.sleep(delay)
        await self.check_all_devices_for_ota()

    def _expire_stale_provider_images(
        self, provider: zigpy.ota.providers.BaseOtaProvider
    ) -> None:
        """Drop a provider's cached images after a prolonged outage.

        A provider that cannot be reached cannot withdraw images either, so
        images from a provider whose index has not been successfully refreshed
        for a long time are revoked instead of being offered indefinitely.
        """
        images = self._image_cache.get(provider)
        if not images:
            return

        now = datetime.datetime.now(datetime.UTC)

        if now - provider._index_last_success <= provider.STALE_INDEX_EXPIRATION_TIME:
            return

        _LOGGER.warning(
            "Provider %s has been unreachable for over %s, dropping its"
            " %d cached images",
            provider,
            provider.STALE_INDEX_EXPIRATION_TIME,
            len(images),
        )
        del self._image_cache[provider]
        self._prune_firmware_cache()

    def _prune_firmware_cache(self) -> None:
        """Drop downloaded firmware no longer referenced by any provider index."""
        live_keys = {
            meta.firmware_cache_key
            for index in self._image_cache.values()
            for meta in index
        }

        for key in list(self._firmware_cache):
            if key not in live_keys:
                _LOGGER.debug("Dropping unreferenced cached firmware: %s", key)
                del self._firmware_cache[key]

    def _get_cached_firmware(
        self, metadata: zigpy.ota.providers.BaseOtaImageMetadata
    ) -> BaseOTAImage | None:
        """Look up downloaded firmware for the given metadata."""
        key = metadata.firmware_cache_key
        firmware = self._firmware_cache.pop(key, None)

        if firmware is None:
            return None

        # Re-insert to mark the firmware as most-recently-used
        self._firmware_cache[key] = firmware

        return firmware

    def _store_firmware(
        self,
        metadata: zigpy.ota.providers.BaseOtaImageMetadata,
        firmware: BaseOTAImage,
    ) -> None:
        """Cache downloaded firmware for the given metadata."""
        _LOGGER.debug("Caching firmware for %s", metadata)
        key = metadata.firmware_cache_key
        self._firmware_cache.pop(key, None)
        self._firmware_cache[key] = firmware

        # Evict the least-recently-used firmware once the cache grows beyond
        # the size limit, always keeping at least the newest entry
        while (
            len(self._firmware_cache) > 1
            and sum(fw.header.image_size for fw in self._firmware_cache.values())
            > FIRMWARE_CACHE_SIZE_LIMIT
        ):
            evicted_key = next(iter(self._firmware_cache))
            _LOGGER.debug(
                "Evicting cached firmware over the size limit: %s", evicted_key
            )
            del self._firmware_cache[evicted_key]

    @zigpy.util.combine_concurrent_calls
    async def _fetch_firmware(
        self, metadata: zigpy.ota.providers.BaseOtaImageMetadata
    ) -> BaseOTAImage:
        """Download and validate an OTA image."""

        async with asyncio_timeout(OTA_FETCH_TIMEOUT):
            return await metadata.fetch()

    async def get_ota_images(
        self,
        device: zigpy.device.Device,
        query_cmd: QueryNextImageCommand,
    ) -> OtaImagesResult:
        """Get OTA images compatible with the device."""
        # Only consider providers that are compatible with the device
        compatible_providers = [
            p for p in self._providers if p.compatible_with_device(device)
        ]

        # Refresh the index of every provider whose cache expired, concurrently:
        # one slow or unreachable provider should not delay the others
        await asyncio.gather(
            *(self._refresh_provider_index(p) for p in compatible_providers)
        )

        # Merge the cached index metadata of all compatible providers and pair
        # each image with its downloaded firmware, if any
        metadata: set[zigpy.ota.providers.BaseOtaImageMetadata] = set()

        for provider in compatible_providers:
            metadata |= self._image_cache.get(provider, set())

        # Find all superficially compatible images. Note that if an image's contents
        # are unknown and its metadata does not describe hardware compatibility, we will
        # still download in the next step to double check, in case the file itself does.
        candidates = sorted(
            [
                img
                for img in (
                    OtaImageWithMetadata(
                        metadata=meta, firmware=self._get_cached_firmware(meta)
                    )
                    for meta in metadata
                )
                if img.check_compatibility(device, query_cmd)
            ],
            key=lambda img: img.version,
        )

        upgrades = {
            img.metadata: img
            for img in candidates
            if img.check_version(query_cmd.current_file_version)
        }
        downgrades = {
            img.metadata: img for img in candidates if img.metadata not in upgrades
        }

        # Only download upgrade images from untrusted providers; trusted providers have
        # complete metadata so we can defer the download until install time. Images
        # sharing a fetch identity (e.g. differing only in release notes) are
        # downloaded once.
        undownloaded: dict[typing.Hashable, list[OtaImageWithMetadata]] = {}

        for img in upgrades.values():
            if img.firmware is None and not img.metadata.trusted:
                undownloaded.setdefault(img.metadata.firmware_cache_key, []).append(img)

        # Fetch all the candidates that are missing from the cache
        results = await asyncio.gather(
            *(
                self._fetch_firmware(group[0].metadata)
                for group in undownloaded.values()
            ),
            return_exceptions=True,
        )

        for group, result in zip(undownloaded.values(), results, strict=True):
            if isinstance(result, BaseException):
                _LOGGER.debug(
                    "Failed to download image, ignoring: %s",
                    group[0].metadata,
                    exc_info=result,
                )
                for img in group:
                    upgrades.pop(img.metadata)
                continue

            self._store_firmware(group[0].metadata, result)

            for img in group:
                img = img.replace(firmware=result)

                if not img.check_compatibility(device, query_cmd):
                    # Ignore images that become incompatible once downloaded
                    del upgrades[img.metadata]
                else:
                    upgrades[img.metadata] = img

        await self._remove_colliding_images(upgrades)

        return OtaImagesResult(
            upgrades=tuple(
                sorted(
                    upgrades.values(),
                    key=lambda img: (img.version, img.specificity),
                    reverse=True,
                )
            ),
            downgrades=tuple(
                sorted(
                    downgrades.values(),
                    key=lambda img: (img.version, img.specificity),
                    reverse=True,
                )
            ),
        )

    async def _remove_colliding_images(
        self,
        upgrades: dict[zigpy.ota.providers.BaseOtaImageMetadata, OtaImageWithMetadata],
    ) -> None:
        """Remove images with identical versions and specificity but differing contents."""
        # Structure: {(version, specificity): {content_hash: [images]}}
        collisions: defaultdict[
            tuple[int, int], defaultdict[str, list[OtaImageWithMetadata]]
        ] = defaultdict(lambda: defaultdict(list))

        for img in upgrades.values():
            # Untrusted images are always downloaded above and ones that failed
            # to download were already removed; this should never happen.
            assert img.firmware is not None or img.metadata.trusted

            # Calculate content hash from firmware if available, otherwise use metadata
            if img.firmware is not None:
                hasher = hashlib.sha3_256()
                await asyncio.get_running_loop().run_in_executor(
                    None, hasher.update, img.firmware.serialize()
                )
                content_hash = "sha3-256:" + hasher.hexdigest()
            else:
                # Trusted images without a SHA3-256 checksum are dropped when
                # the provider's index is refreshed
                assert img.metadata.checksum is not None
                content_hash = img.metadata.checksum

            collisions[img.version, img.specificity][content_hash].append(img)

        for (version, specificity), buckets in collisions.items():
            # If there are multiple unique hashes, we have a collision
            if len(buckets) < 2:
                continue

            bad_images = []

            for bucket in buckets.values():
                bad_images.extend(bucket)

            _LOGGER.warning(
                "Multiple unique OTA images for version %08X with specificity %d exist."
                " It is not possible to tell which image is correct so all %d of the"
                " colliding images will be ignored.",
                version,
                specificity,
                len(bad_images),
            )
            _LOGGER.debug("Colliding images: %s", bad_images)

            for img in bad_images:
                upgrades.pop(img.metadata)

    async def broadcast_notify(
        self,
        broadcast_address: t.BroadcastAddress = t.BroadcastAddress.ALL_DEVICES,
        jitter: int | None = None,
    ) -> None:
        tsn = self._application.get_sequence()

        command = Ota.ClientCommandDefs.image_notify

        # To avoid flooding huge networks, set the jitter such that we will probably
        # have a fixed number of devices checking in at once. All devices should
        # eventually check in, just not every time.
        if jitter is None:
            num_devices = len(self._application.devices)
            ratio = MAX_DEVICES_CHECKING_IN_PER_BROADCAST / max(1, num_devices)
            jitter = int(100 * min(max(0.0, ratio), 1.0))

        hdr, request = Ota._create_request(
            self=None,
            general=False,
            command_id=command.id,
            schema=command.schema,
            tsn=tsn,
            disable_default_response=True,
            direction=foundation.Direction.Server_to_Client,
            args=(),
            kwargs={
                "payload_type": Ota.ImageNotifyCommand.PayloadType.QueryJitter,
                "query_jitter": jitter,
            },
        )

        # Broadcast
        await self._application.send_packet(
            t.ZigbeePacket(
                src=t.AddrModeAddress(
                    addr_mode=t.AddrMode.NWK,
                    address=self._application.state.node_info.nwk,
                ),
                src_ep=1,
                dst=t.AddrModeAddress(
                    addr_mode=t.AddrMode.Broadcast,
                    address=broadcast_address,
                ),
                dst_ep=0xFF,
                tsn=tsn,
                profile_id=zigpy.profiles.zha.PROFILE_ID,
                cluster_id=Ota.cluster_id,
                data=t.SerializableBytes(hdr.serialize() + request.serialize()),
                tx_options=t.TransmitOptions.NONE,
                radius=30,
            )
        )
