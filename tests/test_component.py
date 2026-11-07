"""Tests for the CEDA component: defs building and in-process execution.

The ceda-client boundary is stubbed by the ``fake_ceda`` fixture
(``conftest.py``): jobs run for real in-process against a ``FakeClient``
that records calls and returns canned manifests and result batches.
"""

import re
from pathlib import Path, PosixPath

import dagster as dg
import pytest as pt
import pytest_mock as ptm
import upath as up
import yaml as pyyaml
from ceda_client.client import (
    DownloadResult,
    FileManifest,
    ResultBatch,
)
from conftest import BASE_URL, DATA_DIR, make_file, make_resource

from dagster_ceda import CEDAComponent, CEDAEntity

DATA_FILES = Path(DATA_DIR)


def raw_compute_fn(op: dg.OpDefinition):
    """The raw compute function of an op.

    ``op.compute_fn`` is a decorator wrapper that is not directly
    callable; the function it wraps is what Dagster invokes.
    """
    # compute_fn is typed as a bare callable, but is a wrapper at runtime
    return op.compute_fn.decorated_fn  # pyright: ignore[reportFunctionMemberAccess]


@pt.fixture
def load_context() -> dg.ComponentLoadContext:
    return dg.ComponentTree.for_test().load_context


# ---------------------------------------------------------------------------
# CEDAEntity
# ---------------------------------------------------------------------------


def test_entity_resolves_fields():
    # str values are the yaml form: path and pattern are coerced to their
    # resolved types on validation
    entity = CEDAEntity.model_validate(
        {
            "name": "files",
            "path": DATA_DIR,
            "description": "files in data/files",
            "pattern": "^alpha",
        }
    )
    assert entity.path == DATA_FILES
    assert isinstance(entity.pattern, re.Pattern)
    assert entity.pattern.pattern == "^alpha"
    assert entity.description == "files in data/files"
    assert entity.partition_keys is None
    assert entity.extension is None


# ---------------------------------------------------------------------------
# CEDAComponent construction
# ---------------------------------------------------------------------------


def test_component_defaults():
    component = CEDAComponent(name="dataset", path=DATA_FILES, entities=[])
    assert component.key_prefix == "ceda"
    assert component.ceda_resource_key == "ceda"


@pt.mark.parametrize(
    ("raw", "expected_key"),
    [
        ("ceda", ["ceda", "dataset"]),
        ("a/b", ["a/b", "dataset"]),
        (["x", "y"], ["x", "y", "dataset"]),
    ],
)
def test_external_asset_key_prefix(
    load_context: dg.ComponentLoadContext,
    raw: str | list[str],
    expected_key: list[str],
):
    component = CEDAComponent(
        name="dataset",
        path=DATA_FILES,
        key_prefix=raw,
        entities=[],
    )
    spec = component.make_external_asset(load_context)
    assert spec.key == dg.AssetKey(expected_key)


# ---------------------------------------------------------------------------
# Definition building
# ---------------------------------------------------------------------------


def test_external_asset_spec(load_context: dg.ComponentLoadContext):
    component = CEDAComponent(name="dataset", path=DATA_FILES, entities=[])
    spec = component.make_external_asset(load_context)
    assert spec.key == dg.AssetKey(["ceda", "dataset"])
    assert "source" in spec.kinds
    assert spec.metadata["ceda_path"] == dg.MetadataValue.path(DATA_DIR)


def test_make_asset(load_context: dg.ComponentLoadContext):
    component = CEDAComponent(
        name="dataset",
        path=DATA_FILES,
        entities=[CEDAEntity(name="alpha", path=Path("sub"))],
    )
    external = component.make_external_asset(load_context)
    asset = component.make_asset(load_context, external, component.entities[0])
    assert asset.key == dg.AssetKey(["ceda", "dataset", "alpha"])
    assert asset.partitions_def is None
    assert asset.metadata_by_key[asset.key]["ceda_path"] == (
        dg.MetadataValue.path(f"{DATA_DIR}/sub")
    )


def test_make_asset_check(load_context: dg.ComponentLoadContext):
    component = CEDAComponent(
        name="dataset",
        path=DATA_FILES,
        entities=[CEDAEntity(name="alpha", path=Path("sub"))],
    )
    external = component.make_external_asset(load_context)
    asset = component.make_asset(load_context, external, component.entities[0])
    check = component.make_asset_check(asset)
    assert check.check_key == dg.AssetCheckKey(
        asset_key=dg.AssetKey(["ceda", "dataset", "alpha"]),
        name="result_batch_has_no_failures",
    )
    spec = next(iter(check.check_specs))
    assert spec.blocking is True
    assert spec.partitions_def is asset.partitions_def


def test_component_defs_structure():
    component = CEDAComponent(
        name="ceda-test",
        path=DATA_FILES,
        entities=[CEDAEntity(name="alpha", path=Path("sub"))],
    )
    defs = dg.component_defs(component=component)
    specs = [a for a in defs.assets or () if isinstance(a, dg.AssetSpec)]
    assets = [
        a for a in defs.assets or () if isinstance(a, dg.AssetsDefinition)
    ]
    assert len(specs) == 1
    assert specs[0].key == dg.AssetKey(["ceda", "ceda-test"])
    assert "source" in specs[0].kinds
    assert len(assets) == 1
    assert assets[0].key == dg.AssetKey(["ceda", "ceda-test", "alpha"])
    assert assets[0].partitions_def is None
    checks = list(defs.asset_checks or ())
    assert len(checks) == 1
    spec = next(iter(checks[0].check_specs))
    assert spec.name == "result_batch_has_no_failures"
    assert spec.asset_key == dg.AssetKey(["ceda", "ceda-test", "alpha"])
    assert spec.blocking is True


# ---------------------------------------------------------------------------
# Partition resolution
# ---------------------------------------------------------------------------


def test_resolve_partitions_def():
    component = CEDAComponent(
        name="dataset",
        path=DATA_FILES,
        partition_keys=["a", "b"],
        entities=[
            CEDAEntity(name="inherited", path=Path("inherited")),
            CEDAEntity(name="own", path=Path("own"), partition_keys=["c"]),
        ],
    )
    inherited = component.resolve_partitions_def(component.entities[0])
    assert inherited is not None
    assert list(inherited.get_partition_keys()) == ["a", "b"]

    own = component.resolve_partitions_def(component.entities[1])
    assert own is not None
    assert list(own.get_partition_keys()) == ["c"]

    bare = CEDAComponent(
        name="dataset",
        path=DATA_FILES,
        entities=[CEDAEntity(name="z", path=Path("z"))],
    )
    assert bare.resolve_partitions_def(bare.entities[0]) is None


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def test_get_asset_metadata():
    component = CEDAComponent(
        name="dataset",
        path=DATA_FILES,
        entities=[CEDAEntity(name="alpha", path=Path("sub"))],
    )
    metadata = component.get_asset_metadata(component.entities[0])
    assert metadata["ceda_path"] == dg.MetadataValue.path(f"{DATA_DIR}/sub")


def test_get_output_metadata(mocker: ptm.MockFixture):
    component = CEDAComponent(name="dataset", path=DATA_FILES, entities=[])
    alpha = make_file("alpha.nc")
    corrupt = make_file("corrupt.nc")
    nomd5 = make_file("nomd5.dat")
    manifest = FileManifest(
        files=(alpha, corrupt, nomd5),
        path=DATA_DIR,
        extension=".nc",
        pattern="^x",
    )
    target = up.UPath("/storage")
    batch = ResultBatch(
        results=(
            DownloadResult.succeed(alpha, target / "alpha.nc"),
            DownloadResult.skip(corrupt, target / "corrupt.nc"),
            DownloadResult.fail(nomd5, target / "nomd5.dat", OSError("boom")),
        )
    )
    context = mocker.MagicMock()
    metadata = component.get_output_metadata(context, manifest, batch, target)
    assert metadata["ceda_path"] == dg.MetadataValue.path(DATA_DIR)
    assert metadata["storage_path"] == dg.MetadataValue.path("/storage")
    assert metadata["SUCCESS"] == dg.MetadataValue.int(1)
    assert metadata["SKIPPED"] == dg.MetadataValue.int(1)
    assert metadata["FAILED"] == dg.MetadataValue.int(1)
    assert metadata["extension"] == dg.MetadataValue.text(".nc")
    assert metadata["pattern"] == dg.MetadataValue.text("^x")


def test_get_output_metadata_no_filters(mocker: ptm.MockFixture):
    component = CEDAComponent(name="dataset", path=DATA_FILES, entities=[])
    manifest = FileManifest(files=(), path=DATA_DIR)
    target = up.UPath("/storage")
    batch = ResultBatch(results=())
    context = mocker.MagicMock()
    metadata = component.get_output_metadata(context, manifest, batch, target)
    assert "extension" not in metadata
    assert "pattern" not in metadata
    assert metadata["SUCCESS"] == dg.MetadataValue.int(0)
    assert metadata["SKIPPED"] == dg.MetadataValue.int(0)
    assert metadata["FAILED"] == dg.MetadataValue.int(0)


def test_get_output_metadata_compiled_pattern(mocker: ptm.MockFixture):
    component = CEDAComponent(name="dataset", path=DATA_FILES, entities=[])
    manifest = FileManifest(files=(), path=DATA_DIR, pattern=re.compile("^x"))
    target = up.UPath("/storage")
    batch = ResultBatch(results=())
    context = mocker.MagicMock()
    metadata = component.get_output_metadata(context, manifest, batch, target)
    assert metadata["pattern"] == dg.MetadataValue.text("^x")


# ---------------------------------------------------------------------------
# Ops (stubbed execution context)
# ---------------------------------------------------------------------------


def test_manifest_op_partition_path(
    load_context: dg.ComponentLoadContext, mocker: ptm.MockFixture
):
    component = CEDAComponent(
        name="dataset",
        path=DATA_FILES,
        entities=[CEDAEntity(name="alpha", path=Path("sub"))],
    )
    op = component.make_manifest_op(load_context, component.entities[0])
    assert op.name == "alpha_manifest_op"
    assert op.required_resource_keys == frozenset({"ceda"})

    manifest = FileManifest(files=(), path=f"{DATA_DIR}/sub")
    client = mocker.MagicMock()
    client.get_files.return_value = manifest
    resource = mocker.MagicMock()
    resource.get_client.return_value.__enter__.return_value = client
    context = mocker.MagicMock()
    context.has_partition_key = True
    context.partition_key = "2025-01"
    context.resources.ceda = resource

    result = raw_compute_fn(op)(context)
    assert result is manifest
    client.get_files.assert_called_once_with(
        f"{DATA_DIR}/sub/2025-01", extension=None, pattern=None
    )
    context.add_output_metadata.assert_called_once_with({"n_files": 0})


def test_manifest_op_filter_precedence(
    load_context: dg.ComponentLoadContext, mocker: ptm.MockFixture
):
    component = CEDAComponent(
        name="dataset",
        path=DATA_FILES,
        extension=".nc",
        pattern=re.compile("^component"),
        entities=[
            CEDAEntity(
                name="alpha",
                path=Path("sub"),
                extension=".dat",
                pattern=re.compile("^entity"),
            )
        ],
    )
    op = component.make_manifest_op(load_context, component.entities[0])

    client = mocker.MagicMock()
    client.get_files.return_value = FileManifest(files=(), path=DATA_DIR)
    resource = mocker.MagicMock()
    resource.get_client.return_value.__enter__.return_value = client
    context = mocker.MagicMock()
    context.has_partition_key = False
    context.resources.ceda = resource

    raw_compute_fn(op)(context)
    client.get_files.assert_called_once_with(
        f"{DATA_DIR}/sub",
        extension=".dat",
        pattern=re.compile("^entity"),
    )


def test_manifest_op_inherits_component_filters(
    load_context: dg.ComponentLoadContext, mocker: ptm.MockFixture
):
    component = CEDAComponent(
        name="dataset",
        path=DATA_FILES,
        extension=".nc",
        entities=[CEDAEntity(name="alpha", path=Path("sub"))],
    )
    op = component.make_manifest_op(load_context, component.entities[0])

    client = mocker.MagicMock()
    client.get_files.return_value = FileManifest(files=(), path=DATA_DIR)
    resource = mocker.MagicMock()
    resource.get_client.return_value.__enter__.return_value = client
    context = mocker.MagicMock()
    context.has_partition_key = False
    context.resources.ceda = resource

    raw_compute_fn(op)(context)
    client.get_files.assert_called_once_with(
        f"{DATA_DIR}/sub", extension=".nc", pattern=None
    )


def test_mirror_op_downloads_manifest(
    load_context: dg.ComponentLoadContext, mocker: ptm.MockFixture
):
    component = CEDAComponent(
        name="dataset",
        path=DATA_FILES,
        entities=[CEDAEntity(name="alpha", path=Path("sub"))],
    )
    op = component.make_mirror_op(load_context, component.entities[0])
    assert op.name == "alpha_mirror_op"

    alpha = make_file("alpha.nc")
    manifest = FileManifest(files=(alpha,), path=DATA_DIR)
    target = up.UPath("/out")
    batch = ResultBatch(
        results=(DownloadResult.succeed(alpha, target / "alpha.nc"),)
    )

    client = mocker.MagicMock()
    client.download_multi.return_value = batch
    resource = mocker.MagicMock()
    resource.base_dir = "/out"
    resource.get_client.return_value.__enter__.return_value = client
    context = mocker.MagicMock()
    context.resources.ceda = resource

    result = raw_compute_fn(op)(context, manifest)
    assert result is batch
    client.download_multi.assert_called_once_with(
        files=(alpha,), target=target, mirror_dirs=True
    )
    context.add_output_metadata.assert_called_once()


# ---------------------------------------------------------------------------
# End-to-end runs with the ceda-client boundary stubbed
# ---------------------------------------------------------------------------


def test_component_job_downloads_and_check_passes(
    fake_ceda, tmp_path: PosixPath
):
    alpha = make_file("alpha.nc")
    beta = make_file("beta.nc")
    manifest = FileManifest(files=(alpha, beta), path=DATA_DIR)
    target = up.UPath(tmp_path)
    batch = ResultBatch(
        results=(
            DownloadResult.succeed(alpha, target / "alpha.nc"),
            DownloadResult.succeed(beta, target / "beta.nc"),
        )
    )
    client = fake_ceda(manifest, batch)

    component = CEDAComponent(
        name="ceda-test",
        path=DATA_FILES,
        entities=[CEDAEntity(name="nc", path=Path("."), extension=".nc")],
    )
    resource = make_resource(BASE_URL, tmp_path)
    defs = dg.component_defs(component=component, resources={"ceda": resource})
    job = defs.resolve_implicit_global_asset_job_def()
    result = job.execute_in_process()

    assert result.success
    assert client.get_files_calls == [(DATA_DIR, ".nc", None)]
    assert client.download_multi_calls == [((alpha, beta), target, True)]
    materializations = result.asset_materializations_for_node(
        "ceda__ceda_test__nc"
    )
    assert len(materializations) == 1
    assert materializations[0].asset_key == dg.AssetKey(
        ["ceda", "ceda-test", "nc"]
    )
    metadata = materializations[0].metadata
    assert metadata["ceda_path"] == dg.MetadataValue.path(DATA_DIR)
    assert metadata["storage_path"] == dg.MetadataValue.path(
        tmp_path.as_posix()
    )
    assert metadata["SUCCESS"] == dg.MetadataValue.int(2)
    assert metadata["SKIPPED"] == dg.MetadataValue.int(0)
    assert metadata["FAILED"] == dg.MetadataValue.int(0)
    evaluations = result.get_asset_check_evaluations()
    assert len(evaluations) == 1
    assert evaluations[0].check_name == "result_batch_has_no_failures"
    assert evaluations[0].passed


def test_component_check_fails_on_failed_downloads(
    fake_ceda, tmp_path: PosixPath
):
    # missing.nc is listed but its download fails, so the batch contains
    # one failure and the blocking check must fail the run.
    missing = make_file("missing.nc")
    manifest = FileManifest(files=(missing,), path=DATA_DIR)
    target = up.UPath(tmp_path)
    batch = ResultBatch(
        results=(
            DownloadResult.fail(
                missing, target / "missing.nc", OSError("404")
            ),
        )
    )
    fake_ceda(manifest, batch)

    component = CEDAComponent(
        name="ceda-test",
        path=DATA_FILES,
        entities=[CEDAEntity(name="missing", path=Path("."))],
    )
    resource = make_resource(BASE_URL, tmp_path)
    defs = dg.component_defs(component=component, resources={"ceda": resource})
    job = defs.resolve_implicit_global_asset_job_def()
    # a blocking check failure raises by default; return the failed run
    result = job.execute_in_process(raise_on_error=False)

    assert not result.success
    evaluations = result.get_asset_check_evaluations()
    assert len(evaluations) == 1
    evaluation = evaluations[0]
    assert evaluation.check_name == "result_batch_has_no_failures"
    assert not evaluation.passed
    assert evaluation.metadata["failed"].value == 1
    assert evaluation.metadata["failures"].value == [missing.download_url]


def test_component_job_partitioned(fake_ceda, tmp_path: PosixPath):
    gamma = make_file("gamma.txt")
    manifest = FileManifest(files=(gamma,), path=f"{DATA_DIR}/k1")
    target = up.UPath(tmp_path)
    batch = ResultBatch(
        results=(DownloadResult.succeed(gamma, target / "gamma.txt"),)
    )
    client = fake_ceda(manifest, batch)

    component = CEDAComponent(
        name="ceda-test",
        path=DATA_FILES,
        entities=[
            CEDAEntity(
                name="gamma",
                path=Path("."),
                extension=".txt",
                partition_keys=["k1", "k2"],
            )
        ],
    )
    resource = make_resource(BASE_URL, tmp_path)
    defs = dg.component_defs(component=component, resources={"ceda": resource})
    job = defs.resolve_implicit_global_asset_job_def()
    partitions = job.partitions_def
    assert partitions is not None
    assert list(partitions.get_partition_keys()) == ["k1", "k2"]

    result = job.execute_in_process(partition_key="k1")
    assert result.success
    # the partition key is appended to the remote path
    assert client.get_files_calls == [(f"{DATA_DIR}/k1", ".txt", None)]


def test_component_custom_resource_key(fake_ceda, tmp_path: PosixPath):
    gamma = make_file("gamma.txt")
    manifest = FileManifest(files=(gamma,), path=DATA_DIR)
    target = up.UPath(tmp_path)
    batch = ResultBatch(
        results=(DownloadResult.succeed(gamma, target / "gamma.txt"),)
    )
    client = fake_ceda(manifest, batch)

    component = CEDAComponent(
        name="ceda-test",
        path=DATA_FILES,
        ceda_resource_key="my_ceda",
        entities=[CEDAEntity(name="gamma", path=Path("."), extension=".txt")],
    )
    resource = make_resource(BASE_URL, tmp_path)
    defs = dg.component_defs(
        component=component, resources={"my_ceda": resource}
    )
    job = defs.resolve_implicit_global_asset_job_def()
    result = job.execute_in_process()

    assert result.success
    assert len(client.get_files_calls) == 1
    assert len(client.download_multi_calls) == 1


def test_component_missing_resource_key(tmp_path: PosixPath):
    # the component asks for the default "ceda" resource key, but the
    # resource is only provided under another key
    component = CEDAComponent(
        name="ceda-test",
        path=DATA_FILES,
        entities=[CEDAEntity(name="gamma", path=Path("."))],
    )
    resource = make_resource(BASE_URL, tmp_path)
    defs = dg.component_defs(
        component=component, resources={"some_other_key": resource}
    )
    with pt.raises(
        dg.DagsterInvalidDefinitionError, match="resource with key 'ceda'"
    ):
        defs.resolve_implicit_global_asset_job_def()


# ---------------------------------------------------------------------------
# Yaml loading
# ---------------------------------------------------------------------------


def test_component_from_yaml(load_context: dg.ComponentLoadContext):
    # the yaml a user writes in their defs/ directory; the attributes
    # block is parsed and loaded the way dagster's yaml loader does
    yaml_text = f"""\
type: dagster_ceda.CEDAComponent
attributes:
  name: dataset
  path: {DATA_DIR}
  entities:
    - name: alpha
      path: sub
      pattern: ^alpha
      extension: .nc
      partition_keys:
        - "2025-01"
        - "2025-02"
"""
    attributes = pyyaml.safe_load(yaml_text)["attributes"]
    model = CEDAComponent.model_validate(attributes)
    component = CEDAComponent.load(model, load_context)

    assert component.name == "dataset"
    assert component.path == DATA_FILES
    entity = component.entities[0]
    assert entity.path == Path("sub")
    assert entity.pattern == re.compile("^alpha")

    defs = component.build_defs(load_context)
    specs = [a for a in defs.assets or () if isinstance(a, dg.AssetSpec)]
    assets = [
        a for a in defs.assets or () if isinstance(a, dg.AssetsDefinition)
    ]
    assert [spec.key for spec in specs] == [dg.AssetKey(["ceda", "dataset"])]
    assert [asset.key for asset in assets] == [
        dg.AssetKey(["ceda", "dataset", "alpha"])
    ]
    partitions = assets[0].partitions_def
    assert partitions is not None
    assert list(partitions.get_partition_keys()) == ["2025-01", "2025-02"]
