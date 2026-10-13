"""Conservative limits shared by the GUI and sequential evaluation commands."""
import functools
import gc
import os
from pathlib import Path
from i18n import message

MIB = 1024 ** 2
GPU_BUDGET_MB = 3072
MIN_AVAILABLE_RAM_MB = 1536
MIN_FREE_GPU_MB = 1024
MAX_IMAGE_SIZE = 960
MAX_SOURCE_PIXELS = 16_000_000
_gpu_lease = None


class ResourceLimitError(RuntimeError):
    """A workload stopped before exceeding the project's resource limits."""


def available_ram_mb():
    try:
        values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        return int(values['MemAvailable'].split()[0]) // 1024
    except (OSError, KeyError, ValueError):
        # psutil is optional on non-Linux hosts.
        try:
            import psutil
            return psutil.virtual_memory().available // MIB
        except ImportError:
            raise ResourceLimitError('无法读取可用内存，停止加载；请安装 psutil 后重试。')


def _check_ram():
    free = available_ram_mb()
    if free < MIN_AVAILABLE_RAM_MB:
        raise ResourceLimitError(message('可用系统内存仅 {free} MiB，低于 {minimum} MiB；请先关闭其他大程序。', free=free, minimum=MIN_AVAILABLE_RAM_MB))


def process_max_rss_mib():
    """Process lifetime peak RSS in MiB, with native platform units."""
    if os.name == 'nt':
        import psutil
        return psutil.Process().memory_info().peak_wset / MIB
    import resource
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (MIB if sys.platform == 'darwin' else 1024)


def _acquire_gpu_lease():
    global _gpu_lease
    if _gpu_lease is not None:
        return
    lock_path = Path(__file__).resolve().parent / '.cache' / 'gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            # Windows locks a byte range. Retain the first byte while writing
            # diagnostic metadata so competing processes lock the same range.
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b'\0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        import errno
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            raise ResourceLimitError('另一个检测进程正在使用 GPU；请先在界面释放模型或结束该检测任务。') from exc
        raise
    try:
        handle.seek(1 if os.name == 'nt' else 0)
        handle.truncate()
        handle.write(str(os.getpid()).encode('ascii'))
        handle.flush()
    except Exception:
        handle.close()
        raise
    _gpu_lease = handle


def release_gpu_lease():
    global _gpu_lease
    if _gpu_lease is not None:
        _gpu_lease.close()
        _gpu_lease = None


def prepare_device(device):
    """Reserve one detector process and cap PyTorch allocations to 3 GiB."""
    _check_ram()
    target = str(device).lower()
    if target == 'cpu':
        return 'cpu'
    if target not in ('cuda', 'cuda:0'):
        raise ValueError('安全模式只允许 CPU 或单张 CUDA:0 显卡')
    _acquire_gpu_lease()
    try:
        import torch
        if not torch.cuda.is_available():
            raise ResourceLimitError('CUDA_UNAVAILABLE')
        free, total = torch.cuda.mem_get_info(0)
        if free < (GPU_BUDGET_MB + MIN_FREE_GPU_MB) * MIB:
            raise ResourceLimitError('GPU 空闲显存不足 4 GiB，停止加载模型。')
        torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_BUDGET_MB * MIB / total), 0)
        return 'cuda:0'
    except Exception:
        release_gpu_lease()
        raise


def preflight_inference(device, imgsz=640, frame_shape=None):
    try:
        valid_size = not isinstance(imgsz, bool) and int(imgsz) == imgsz and 32 <= int(imgsz) <= MAX_IMAGE_SIZE
    except (TypeError, ValueError, OverflowError):
        valid_size = False
    if not valid_size:
        raise ResourceLimitError(message('安全模式要求推理尺寸在 32–{maximum} 之间。', maximum=MAX_IMAGE_SIZE))
    if frame_shape is not None and int(frame_shape[0]) * int(frame_shape[1]) > MAX_SOURCE_PIXELS:
        raise ResourceLimitError('输入图像超过 1600 万像素，请先缩小图像。')
    _check_ram()
    if str(device).lower().startswith('cuda'):
        import torch
        free, _ = torch.cuda.mem_get_info(0)
        if free < MIN_FREE_GPU_MB * MIB:
            raise ResourceLimitError('GPU 空闲显存低于 1 GiB，已停止当前推理。')


def handle_cuda_oom(exc):
    message = str(exc).lower()
    is_oom = ('out of memory' in message and ('cuda' in message or 'gpu' in message))
    if is_oom:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return is_oom


def guarded_inference(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            if handle_cuda_oom(exc):
                raise ResourceLimitError('达到显存上限，当前推理已停止；请释放模型或减小输入尺寸。') from exc
            raise
    return wrapped
