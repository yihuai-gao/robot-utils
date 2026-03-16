import time


def wait_until(monotonic_time: float, sleep_threshold: float = 0.001):
    """
    Wait until the monotonic time is greater than the target time.
    """
    while time.monotonic() < monotonic_time - sleep_threshold:
        time.sleep(min(sleep_threshold, monotonic_time - time.monotonic()))

    # If the remaining time is less than the sleep threshold, will just wait without potential interrupt
    while time.monotonic() < monotonic_time:
        pass
