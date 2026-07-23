"""TTLCache 单元测试"""

from datetime import datetime, timedelta
from AIShareTxt.utils.cache import TTLCache


class FakeClock:
    """可手动推进的假时钟，用于 TTL 测试。"""

    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


def test_set_then_get_within_ttl():
    clock = FakeClock(datetime(2026, 7, 22, 10, 0, 0))
    cache = TTLCache(now_fn=clock)

    cache.set("k", {"v": 1})
    assert cache.get("k", ttl=timedelta(seconds=30)) == {"v": 1}


def test_get_missing_returns_none():
    cache = TTLCache(now_fn=lambda: datetime(2026, 7, 22))
    assert cache.get("nope", ttl=timedelta(seconds=10)) is None


def test_expired_entry_returns_none():
    clock = FakeClock(datetime(2026, 7, 22, 10, 0, 0))
    cache = TTLCache(now_fn=clock)

    cache.set("k", "value")
    clock.advance(31)
    assert cache.get("k", ttl=timedelta(seconds=30)) is None


def test_entry_exactly_at_ttl_is_expired():
    """边界：now - set_at == ttl 视为过期（使用 >= 比较）。"""
    clock = FakeClock(datetime(2026, 7, 22, 10, 0, 0))
    cache = TTLCache(now_fn=clock)

    cache.set("k", "value")
    clock.advance(30)
    assert cache.get("k", ttl=timedelta(seconds=30)) is None


def test_overwrite_existing_key():
    cache = TTLCache(now_fn=lambda: datetime(2026, 7, 22))
    cache.set("k", "v1")
    cache.set("k", "v2")
    assert cache.get("k", ttl=timedelta(seconds=10)) == "v2"


def test_invalidate_key():
    cache = TTLCache(now_fn=lambda: datetime(2026, 7, 22))
    cache.set("k", "v")
    cache.invalidate("k")
    assert cache.get("k", ttl=timedelta(seconds=10)) is None


def test_invalidate_missing_key_is_noop():
    cache = TTLCache(now_fn=lambda: datetime(2026, 7, 22))
    cache.invalidate("never_existed")  # should not raise


def test_clear_all():
    cache = TTLCache(now_fn=lambda: datetime(2026, 7, 22))
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert cache.get("a", ttl=timedelta(seconds=10)) is None
    assert cache.get("b", ttl=timedelta(seconds=10)) is None


def test_different_ttls_for_same_value():
    """同一缓存项可用不同 TTL 读多次：短 TTL 过期但长 TTL 仍命中。"""
    clock = FakeClock(datetime(2026, 7, 22, 10, 0, 0))
    cache = TTLCache(now_fn=clock)
    cache.set("k", "value")

    clock.advance(20)
    assert cache.get("k", ttl=timedelta(seconds=10)) is None  # 短 TTL 已过期
    assert cache.get("k", ttl=timedelta(seconds=60)) == "value"  # 长 TTL 仍有效
