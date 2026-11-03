import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Coroutine


class AsyncRunner:
    """Run coroutines on a background event loop so the Qt thread is never blocked."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="gui-async-runner", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self._loop.close()

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future:
        """Schedule a coroutine and return immediately."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self, timeout: float = 2.0) -> None:
        """Cancel pending work and stop the background loop."""
        if not self._thread.is_alive():
            return

        async def cancel_pending() -> None:
            current = asyncio.current_task()
            pending = [task for task in asyncio.all_tasks() if task is not current]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(cancel_pending(), self._loop).result(timeout)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout)
