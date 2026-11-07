from typing import Final

import ceda_client as cc
import dagster as dg
import upath as up

from dagster_ceda.resolved import (
    ResolvedKeyPrefix,
    ResolvedPath,
    ResolvedPattern,
)
from dagster_ceda.resources import CEDAResource

DEFAULT_CEDA_RESOURCE_KEY: Final = "ceda"


class CEDAEntity(dg.Model, dg.Resolvable):
    name: str
    key_prefix: ResolvedKeyPrefix = "ceda"
    path: ResolvedPath
    description: str | None = None
    partition_keys: list[str] | None = None
    extension: str | None = None
    pattern: ResolvedPattern | None = None


class CEDAComponent(dg.Component, CEDAEntity):
    ceda_resource_key: str = DEFAULT_CEDA_RESOURCE_KEY
    entities: list[CEDAEntity]

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        external = self.make_external_asset(context)
        assets = [
            self.make_asset(context, external, entity)
            for entity in self.entities
        ]
        checks = [self.make_asset_check(asset) for asset in assets]
        return dg.Definitions(assets=[external, *assets], asset_checks=checks)

    def get_asset_spec(
        self, context: dg.ComponentLoadContext, entity: CEDAEntity
    ) -> dg.AssetSpec:
        return dg.AssetSpec(
            key=dg.AssetKey(entity.name).with_prefix(entity.key_prefix),
            metadata={
                "ceda_path": dg.MetadataValue.path(self.path),
            },
            kinds={"source"},
        )

    def make_external_asset(
        self, context: dg.ComponentLoadContext
    ) -> dg.AssetSpec:
        return dg.AssetSpec(
            key=dg.AssetKey(self.name).with_prefix(self.key_prefix),
            metadata={
                "ceda_path": dg.MetadataValue.path(self.path),
            },
            kinds={"source"},
        )

    def make_manifest_op(
        self, context: dg.ComponentLoadContext, entity: CEDAEntity
    ) -> dg.OpDefinition:

        @dg.op(
            name=f"{entity.name}_manifest_op",
            description="Generates a manifest describing files in a remote directory.",
            required_resource_keys={self.ceda_resource_key},
            ins={"source": dg.In(dg.Nothing)},
        )
        def _op(context: dg.OpExecutionContext) -> cc.FileManifest:
            path = self.path.joinpath(entity.path)
            if context.has_partition_key:
                path = path.joinpath(context.partition_key)

            ceda: CEDAResource = getattr(
                context.resources, self.ceda_resource_key
            )
            with ceda.get_client() as client:
                manifest = client.get_files(
                    path.as_posix(),
                    extension=entity.extension or self.extension,
                    pattern=entity.pattern or self.pattern,
                )

            context.add_output_metadata({"n_files": len(manifest)})
            return manifest

        return _op

    def make_mirror_op(
        self, context: dg.ComponentLoadContext, entity: CEDAEntity
    ):
        @dg.op(
            name=f"{entity.name}_mirror_op",
            description="Mirrors remote files to a storage directory.",
            required_resource_keys={self.ceda_resource_key},
        )
        def _op(
            context: dg.OpExecutionContext, manifest: cc.FileManifest
        ) -> cc.ResultBatch:
            ceda: CEDAResource = getattr(
                context.resources, self.ceda_resource_key
            )
            target = up.UPath(ceda.base_dir)
            with ceda.get_client() as client:
                batch = client.download_multi(
                    files=manifest.files,
                    target=target,
                    mirror_dirs=True,
                )
            context.add_output_metadata(
                self.get_output_metadata(context, manifest, batch, target)
            )
            return batch

        return _op

    def make_asset(
        self,
        context: dg.ComponentLoadContext,
        external: dg.AssetSpec,
        entity: CEDAEntity,
    ) -> dg.AssetsDefinition:

        get_manifest = self.make_manifest_op(context, entity)
        download_files = self.make_mirror_op(context, entity)

        @dg.graph_asset(
            key=dg.AssetKey(entity.name).with_prefix(external.key.path),
            description=entity.description,
            metadata=self.get_asset_metadata(entity),
            ins={
                "source": dg.AssetIn(key=external.key, dagster_type=dg.Nothing)
            },
            partitions_def=self.resolve_partitions_def(entity),
        )
        def _asset(source):
            return download_files(get_manifest(source))

        return _asset

    def make_asset_check(
        self, asset: dg.AssetsDefinition
    ) -> dg.AssetChecksDefinition:
        @dg.asset_check(
            asset=asset,
            name="result_batch_has_no_failures",
            blocking=True,
            partitions_def=asset.partitions_def,
        )
        def _check(
            context: dg.AssetCheckExecutionContext, batch: cc.ResultBatch
        ) -> dg.AssetCheckResult:
            passed = len(batch.failed) == 0
            if passed:
                return dg.AssetCheckResult(passed=True)
            return dg.AssetCheckResult(
                passed=False,
                metadata={
                    "failed": dg.MetadataValue.int(len(batch.failed)),
                    "failures": [
                        result.file.download_url for result in batch.failed
                    ],
                },
            )

        return _check

    def resolve_partitions_def(
        self, entity: CEDAEntity
    ) -> dg.PartitionsDefinition | None:
        keys = entity.partition_keys or self.partition_keys
        if keys:
            return dg.StaticPartitionsDefinition(keys)
        return None

    def get_asset_metadata(
        self, entity: CEDAEntity
    ) -> dict[str, dg.MetadataValue]:
        return {
            "ceda_path": dg.MetadataValue.path(self.path / entity.path),
        }

    def get_output_metadata(
        self,
        context: dg.OpExecutionContext,
        manifest: cc.FileManifest,
        batch: cc.ResultBatch,
        target: up.UPath,
    ) -> dict[str, dg.MetadataValue]:
        metadata = {
            "ceda_path": dg.MetadataValue.path(manifest.path),
            "storage_path": dg.MetadataValue.path(target.as_posix()),
            **{
                status.name: dg.MetadataValue.int(batch.counter[status])
                for status in cc.Status
            },
        }
        if extension := manifest.extension:
            metadata.update(extension=dg.MetadataValue.text(extension))
        if pattern := manifest.pattern:
            pattern_str = (
                pattern if isinstance(pattern, str) else pattern.pattern
            )
            metadata.update(pattern=dg.MetadataValue.text(pattern_str))
        return metadata
