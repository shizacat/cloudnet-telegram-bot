
from typing import Optional

from aiohttp import web
from prometheus_client import REGISTRY, CollectorRegistry
from prometheus_client.exposition import _bake_output


async def handle_metrics(request: web.Request) -> web.Response:
    """Handles metrics requests."""
    # return web.Response(text='Hello, world')
    registry: CollectorRegistry = REGISTRY
    # Bake output
    status, headers, output = _bake_output(
        registry,
        accept_header=request.headers.get('Accept'),
        accept_encoding_header=request.headers.get('Accept-Encoding'),
        params=request.query,
        disable_compression=False
    )
    return web.Response(body=output, headers=headers)


async def start_aiohttp_server(
    port: int = 9001, host: Optional[str] = None
) -> None:
    """Starts an HTTP server for prometheus metrics."""
    app = web.Application()
    app.router.add_get('/metrics', handle_metrics)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, port=port, host=host)
    await site.start()

    # To stop serving call AppRunner.cleanup():
    # await runner.cleanup()
