import time
import functools
import inspect
import logging
from typing import Callable, Any, Optional, Tuple, TypeVar, Awaitable

F = TypeVar("F", bound=Callable[..., Any])

def timeit(
    _func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    unit: str = "ms",              # 可选: "s" | "ms" | "us" | "ns"
    threshold: Optional[float] = None,  # 小于阈值则不输出
    return_elapsed: bool = False   # True 时返回 (result, elapsed)
) -> Callable[[F], F]:
    """
    装饰器：测量被装饰函数（同步或异步）的运行时间。

    参数
    ----
    name : 自定义函数名（默认用 func.__qualname__）
    logger : 指定 logger；不传则用 print 输出
    unit : 时间单位，"s"/"ms"/"us"/"ns"
    threshold : 只有当耗时 >= 阈值 时才输出
    return_elapsed : 若为 True，包装后函数返回 (原返回值, 耗时 in unit)

    用法
    ----
    @timeit
    def f(...): ...

    @timeit(unit="us", threshold=100.0)
    def g(...): ...

    @timeit(return_elapsed=True)
    def h(...): -> (result, elapsed)
    """
    unit_map = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    if unit not in unit_map:
        raise ValueError(f"Unsupported unit: {unit}. Choose from {list(unit_map)}")
    factor = unit_map[unit]  # 秒 -> 目标单位 的倍率

    log = logger  # 可能为 None

    def _format_msg(func_name: str, elapsed_s: float) -> str:
        elapsed = elapsed_s * factor
        return f"[timeit] {func_name} took {elapsed:.3f} {unit}"

    def _maybe_log(msg: str) -> None:
        if log is not None:
            log.info(msg)
        else:
            print(msg)

    def _decorator(func: F) -> F:
        fname = name or func.__qualname__

        # 异步函数
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                finally:
                    elapsed_s = time.perf_counter() - start
                    if threshold is None or (elapsed_s * factor) >= threshold:
                        _maybe_log(_format_msg(fname, elapsed_s))
                if return_elapsed:
                    return result, elapsed_s * factor
                return result
            return async_wrapper  # type: ignore[return-value]

        # 同步函数
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                elapsed_s = time.perf_counter() - start
                if threshold is None or (elapsed_s * factor) >= threshold:
                    _maybe_log(_format_msg(fname, elapsed_s))
            if return_elapsed:
                return result, elapsed_s * factor
            return result
        return sync_wrapper  # type: ignore[return-value]

    # 兼容 @timeit 和 @timeit(...)
    if _func is not None and callable(_func):
        return _decorator(_func)  # type: ignore[return-value]
    return _decorator
