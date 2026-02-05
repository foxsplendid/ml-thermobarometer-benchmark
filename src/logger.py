# -*- coding: utf-8 -*-
"""
统一日志模块

功能：
- 控制台输出（INFO 级别，彩色）
- 文件输出（DEBUG 级别，保存到 results/logs/）
- 支持实验 ID 前缀
- 日志轮转（避免磁盘占用过大）

使用方式：
    from src.logger import get_logger

    logger = get_logger(__name__)
    logger.info("开始实验...")
    logger.debug("详细调试信息")
    logger.warning("警告信息")
    logger.error("错误信息")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


# ============================================================
# 日志格式定义
# ============================================================

# 控制台格式（简洁）
CONSOLE_FORMAT = '%(asctime)s | %(levelname)-5s | %(message)s'
CONSOLE_DATE_FORMAT = '%H:%M:%S'

# 文件格式（详细）
FILE_FORMAT = '%(asctime)s | %(levelname)-5s | %(name)s | %(message)s'
FILE_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# ============================================================
# 颜色支持（Windows 兼容）
# ============================================================

class ColorFormatter(logging.Formatter):
    """带颜色的日志格式化器（支持 Windows）"""

    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m',      # 重置
    }

    def __init__(self, fmt: str, datefmt: str, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color and self._supports_color()

    @staticmethod
    def _supports_color() -> bool:
        """检测终端是否支持颜色"""
        # Windows 10+ 支持 ANSI
        if sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # 启用 ANSI 支持
                kernel32.SetConsoleMode(
                    kernel32.GetStdHandle(-11), 7
                )
                return True
            except Exception:
                return False
        # Unix 系统通常支持
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


# ============================================================
# 日志器管理
# ============================================================

# 全局日志目录
_LOG_DIR: Optional[Path] = None
_INITIALIZED: bool = False
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 5


def setup_logging(
    log_dir: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_filename: Optional[str] = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    force: bool = False,
) -> None:
    """
    初始化日志系统

    Parameters
    ----------
    log_dir : str, optional
        日志目录，默认为 results/logs
    console_level : int
        控制台日志级别，默认 INFO
    file_level : int
        文件日志级别，默认 DEBUG
    log_filename : str, optional
        日志文件名，默认按时间生成
    max_bytes : int
        单个日志文件最大大小（字节），超出后滚动
    backup_count : int
        日志滚动保留份数
    force : bool
        是否强制重新初始化日志器
    """
    global _LOG_DIR, _INITIALIZED

    if _INITIALIZED and not force:
        return

    # 确定日志目录
    if log_dir is None:
        try:
            from config import CONFIG
            log_dir = CONFIG.output.log_dir
        except ImportError:
            log_dir = 'results/logs'

    _LOG_DIR = Path(log_dir)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 日志文件名
    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'benchmark_{timestamp}.log'

    log_path = _LOG_DIR / log_filename

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 清除已有处理器
    root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColorFormatter(CONSOLE_FORMAT, CONSOLE_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # 文件处理器（滚动日志，避免单文件过大）
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATE_FORMAT))
    root_logger.addHandler(file_handler)

    _INITIALIZED = True

    # 记录初始化信息
    root_logger.debug(f"日志系统初始化完成，日志文件: {log_path}")


def get_logger(name: str) -> logging.Logger:
    """
    获取命名日志器

    Parameters
    ----------
    name : str
        日志器名称，通常使用 __name__

    Returns
    -------
    logging.Logger
        日志器实例
    """
    # 确保日志系统已初始化
    if not _INITIALIZED:
        setup_logging()

    return logging.getLogger(name)


# ============================================================
# 便捷函数
# ============================================================

def log_experiment_start(exp_id: str, config: dict) -> None:
    """
    记录实验开始

    Parameters
    ----------
    exp_id : str
        实验 ID
    config : dict
        实验配置
    """
    logger = get_logger('experiment')
    logger.info(f"{'='*60}")
    logger.info(f"实验开始: {exp_id}")
    logger.info(f"{'='*60}")
    for key, value in config.items():
        logger.debug(f"  {key}: {value}")


def log_experiment_end(exp_id: str, metrics: dict, elapsed: float) -> None:
    """
    记录实验结束

    Parameters
    ----------
    exp_id : str
        实验 ID
    metrics : dict
        实验指标
    elapsed : float
        耗时（秒）
    """
    logger = get_logger('experiment')
    logger.info(f"实验完成: {exp_id} (耗时 {elapsed:.1f}s)")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


def log_fold_progress(fold_idx: int, n_folds: int, metrics: dict) -> None:
    """
    记录折叠进度

    Parameters
    ----------
    fold_idx : int
        当前折索引（0-based）
    n_folds : int
        总折数
    metrics : dict
        当前折指标
    """
    logger = get_logger('cv')
    rmse = metrics.get('rmse', float('nan'))
    r2 = metrics.get('r2', float('nan'))
    logger.info(f"  Fold {fold_idx + 1}/{n_folds}: RMSE={rmse:.3f}, R²={r2:.4f}")


# ============================================================
# 模块测试
# ============================================================

if __name__ == '__main__':
    print("=== 日志模块测试 ===\n")

    # 初始化（使用临时目录）
    import tempfile
    test_log_dir = tempfile.mkdtemp()
    setup_logging(log_dir=test_log_dir, console_level=logging.DEBUG)

    # 获取日志器
    logger = get_logger(__name__)

    # 测试各级别
    logger.debug("这是 DEBUG 级别消息")
    logger.info("这是 INFO 级别消息")
    logger.warning("这是 WARNING 级别消息")
    logger.error("这是 ERROR 级别消息")

    # 测试实验日志
    log_experiment_start("E07_test", {"model": "ert", "data": "augmented"})
    log_fold_progress(0, 10, {"rmse": 30.5, "r2": 0.934})
    log_fold_progress(1, 10, {"rmse": 31.2, "r2": 0.930})
    log_experiment_end("E07_test", {"T_rmse": 30.85, "T_r2": 0.932}, 12.5)

    print(f"\n日志文件保存于: {test_log_dir}")
    print("✅ 日志模块测试通过！")
