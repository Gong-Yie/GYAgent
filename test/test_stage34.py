import time

from self_cognition.lifecycle import ApplicationLifecycle


class FlakyEventBus:
    def __init__(self) -> None:
        self.calls = 0

    def drain(self) -> tuple[object, ...]:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("transient worker failure")
        return ()


def test_worker_survives_transient_drain_error():
    bus = FlakyEventBus()
    lifecycle = ApplicationLifecycle(
        bus,
        worker_enabled=True,
        worker_poll_interval_seconds=0.01,
    )
    lifecycle.start()
    try:
        deadline = time.monotonic() + 2.5
        while bus.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        lifecycle.stop()

    assert bus.calls >= 2