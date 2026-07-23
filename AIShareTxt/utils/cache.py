#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTL 缓存工具

为高频数据（如全市场快照）提供带过期时间的缓存。
模块级单例 `_global_cache` 跨模块共享，避免重复请求。
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple


class TTLCache:
    """带 TTL（time-to-live）的简单内存缓存。

    每个条目存储 (value, set_at) 二元组；读取时按 `ttl` 判断是否过期。
    时钟通过 `now_fn` 注入，便于测试。
    """

    def __init__(self, now_fn: Callable[[], datetime] = None):
        self._store: Dict[str, Tuple[Any, datetime]] = {}
        self._now_fn = now_fn or datetime.now

    def get(self, key: str, ttl: timedelta) -> Optional[Any]:
        """读取缓存。命中且未过期则返回值，否则返回 None。"""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, set_at = entry
        if self._now_fn() - set_at >= ttl:
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """写入缓存（时间戳取自 now_fn）。"""
        self._store[key] = (value, self._now_fn())

    def invalidate(self, key: str) -> None:
        """显式作废某个键。"""
        self._store.pop(key, None)

    def clear(self) -> None:
        """清空整个缓存。"""
        self._store.clear()


# 模块级单例，跨模块共享
_global_cache = TTLCache()


def get_global_cache() -> TTLCache:
    """获取全局 TTL 缓存单例。"""
    return _global_cache
