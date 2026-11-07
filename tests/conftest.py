"""Shared fixtures and helpers for the dagster-ceda test suite.

The suite stubs dagster-ceda's only boundary with ceda-client:
``CEDAResource.get_client``. ``FakeClient`` records the calls the
component's ops make and returns canned ceda-client data objects, so
jobs run for real in-process without any HTTP server or token cache.
"""

import re
from pathlib import Path, PosixPath
from typing import Self

import pytest as pt
import pytest_mock as ptm
import upath as up
from ceda_client.client import FileManifest, ResultBatch
from ceda_client.schema import File, Location

from dagster_ceda import CEDAResource

USERNAME = "testuser"
PASSWORD = "testpass"
BASE_URL = "http://ceda.example.com/"
DATA_DIR = "data/files"


def make_file(name: str, tree: str = DATA_DIR) -> File:
    """Build a CEDA file entry under the given remote tree."""
    return File(  # pyright: ignore[reportCallIssue]
        path=Path(f"/{tree}/{name}"),
        download_url=f"{BASE_URL}files/{name}",
        location=[Location.DISK],
        md5="md5",
        size=1,
        type="file",  # pyright: ignore[reportCallIssue]
    )


def make_resource(url: str, base_dir: PosixPath) -> CEDAResource:
    return CEDAResource(
        username=USERNAME,
        password=PASSWORD,
        url=url,
        base_dir=base_dir.as_posix(),
    )


class FakeClient:
    """Stand-in for ``ceda_client.Client``.

    Records the calls the component's ops make and returns canned
    manifests and result batches. Usable as a context manager, like
    the real client.
    """

    def __init__(self, manifest: FileManifest, batch: ResultBatch) -> None:
        self.manifest = manifest
        self.batch = batch
        self.get_files_calls: list[
            tuple[str, str | None, str | re.Pattern | None]
        ] = []
        self.download_multi_calls: list[
            tuple[tuple[File, ...], up.UPath, bool]
        ] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_args: object) -> None:
        return None

    def get_files(
        self,
        path: str,
        extension: str | None = None,
        pattern: str | re.Pattern | None = None,
    ) -> FileManifest:
        self.get_files_calls.append((path, extension, pattern))
        return self.manifest

    def download_multi(
        self,
        files: tuple[File, ...],
        target: up.UPath,
        *,
        mirror_dirs: bool = False,
        **_kwargs: object,
    ) -> ResultBatch:
        self.download_multi_calls.append((files, target, mirror_dirs))
        return self.batch


@pt.fixture
def fake_ceda(mocker: ptm.MockFixture):
    """Factory that installs a ``FakeClient`` as the resource's client.

    Returns the installed fake so tests can assert on the calls the
    component's ops made.
    """

    def _fake_ceda(manifest: FileManifest, batch: ResultBatch) -> FakeClient:
        client = FakeClient(manifest, batch)
        mocker.patch.object(CEDAResource, "get_client", return_value=client)
        return client

    return _fake_ceda
