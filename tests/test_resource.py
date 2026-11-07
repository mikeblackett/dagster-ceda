from pathlib import PosixPath

import pytest as pt
from ceda_client.client import CEDA_ENDPOINT_URL, DEFAULT_WORKERS, Client
from conftest import BASE_URL, PASSWORD, USERNAME
from pydantic import ValidationError

from dagster_ceda import CEDAResource


def test_required_config(tmp_path: PosixPath):
    with pt.raises(ValidationError):
        CEDAResource(username=USERNAME)  # pyright: ignore[reportCallIssue]
    resource = CEDAResource(
        username=USERNAME, password=PASSWORD, base_dir=tmp_path.as_posix()
    )
    assert resource.username == USERNAME
    assert resource.url == CEDA_ENDPOINT_URL
    assert resource.max_workers == DEFAULT_WORKERS


def test_get_client_applies_config(tmp_path: PosixPath):
    resource = CEDAResource(
        username=USERNAME,
        password=PASSWORD,
        url=BASE_URL,
        max_workers=4,
        base_dir=tmp_path.as_posix(),
    )
    # client creation is I/O-free, and the client's config must mirror
    # the resource's
    with resource.get_client() as client:
        assert isinstance(client, Client)
        assert client.username == USERNAME
        assert client.url == BASE_URL
        assert client.max_workers == 4
    # the resource is a stateless factory: a fresh client per call
    assert resource.get_client() is not client
