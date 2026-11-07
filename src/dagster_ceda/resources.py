import ceda_client as cc
import dagster as dg
from ceda_client.client import (
    CEDA_ENDPOINT_URL,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_READ_TIMEOUT_SECONDS,
    DEFAULT_WORKERS,
)

__all__ = ["CEDAResource"]


class CEDAResource(dg.ConfigurableResource):
    """Dagster resource that creates CEDA ``Client`` instances.

    Use :meth:`get_client` as a context manager inside ops and assets:

    .. code-block:: python

        @dg.asset
        def my_asset(ceda: CEDAResource):
            with ceda.get_client() as client:
                files = client.get_files("/data/some/dir", extension=".nc")
                client.download_multi(files, target)
    """

    username: str
    password: str
    url: str = CEDA_ENDPOINT_URL
    max_workers: int = DEFAULT_WORKERS
    connect_timeout: float | None = DEFAULT_CONNECT_TIMEOUT_SECONDS
    read_timeout: float | None = DEFAULT_READ_TIMEOUT_SECONDS
    base_dir: str

    def get_client(self) -> cc.Client:
        """Create a CEDA client from this resource's config.

        Use it as a context manager so the HTTP session is closed after use.

        The resource is a stateless factory: ``get_client`` returns a fresh
        ``Client`` per call and the caller scopes its lifetime with the context
        manager. This is deliberate. Holding the client in ``setup_for_execution``
        and ``teardown_after_execution`` would make the client's lifetime dependent
        on how Dagster initializes the resource. Sharing sessions across runs
        would buy little anyway, since ``Client`` creation is I/O-free and
        ``Client`` manages its own worker sessions.
        """
        return cc.Client(
            self.username,
            self.password,
            url=self.url,
            max_workers=self.max_workers,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
        )
