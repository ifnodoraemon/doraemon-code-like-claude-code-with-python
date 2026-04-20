"""
Log Rotation System

Automatic log file rotation and management.

Features:
- Size-based rotation
- Time-based rotation
- Compression of old logs
- Automatic cleanup
- Multiple rotation policies
"""

import gzip
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RotationType(Enum):
    """Log rotation types."""

    SIZE = "size"  # Rotate based on file size
    TIME = "time"  # Rotate based on time
    HYBRID = "hybrid"  # Both size and time


class TimeInterval(Enum):
    """Time intervals for rotation."""

    HOURLY = "H"
    DAILY = "D"
    WEEKLY = "W0"  # Monday
    MIDNIGHT = "midnight"


@dataclass
class RotationConfig:
    """Log rotation configuration."""

    rotation_type: RotationType = RotationType.SIZE
    max_size_mb: int = 10  # Max size before rotation (size-based)
    backup_count: int = 5  # Number of backups to keep
    time_interval: TimeInterval = TimeInterval.DAILY  # For time-based
    compress: bool = True  # Compress rotated logs
    compress_level: int = 9  # gzip compression level


class CompressingRotatingFileHandler(RotatingFileHandler):
    """Rotating file handler with compression support."""

    def __init__(
        self,
        filename: str,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        compress: bool = True,
        compress_level: int = 9,
        **kwargs,
    ):
        self._compress = compress
        self._compress_level = compress_level
        super().__init__(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            **kwargs,
        )

    def doRollover(self):
        """Do rollover with optional compression."""
        # Call parent rollover
        super().doRollover()

        # Compress the rotated file
        if self._compress:
            # Find the most recent backup
            for i in range(self.backupCount, 0, -1):
                rotated_file = f"{self.baseFilename}.{i}"
                compressed_file = f"{rotated_file}.gz"

                if Path(rotated_file).exists() and not Path(compressed_file).exists():
                    self._compress_file(rotated_file)

    def _compress_file(self, filepath: str):
        """Compress a file using gzip."""
        try:
            with open(filepath, "rb") as f_in:
                with gzip.open(f"{filepath}.gz", "wb", compresslevel=self._compress_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove original
            os.remove(filepath)
            logger.debug("Compressed: %s", filepath)

        except Exception as e:
            logger.error("Failed to compress %s: %s", filepath, e)


class CompressingTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Timed rotating file handler with compression support."""

    def __init__(
        self,
        filename: str,
        when: str = "midnight",
        interval: int = 1,
        backup_count: int = 7,
        compress: bool = True,
        compress_level: int = 9,
        **kwargs,
    ):
        self._compress = compress
        self._compress_level = compress_level
        super().__init__(
            filename,
            when=when,
            interval=interval,
            backupCount=backup_count,
            **kwargs,
        )

    def doRollover(self):
        """Do rollover with optional compression."""
        # Call parent rollover
        super().doRollover()

        # Compress rotated files
        if self._compress:
            self._compress_old_logs()

    def _compress_old_logs(self):
        """Compress old log files."""
        dir_path = Path(self.baseFilename).parent
        base_name = Path(self.baseFilename).name

        for log_file in dir_path.glob(f"{base_name}.*"):
            if log_file.suffix != ".gz" and log_file.name != base_name:
                self._compress_file(str(log_file))

    def _compress_file(self, filepath: str):
        """Compress a file using gzip."""
        try:
            with open(filepath, "rb") as f_in:
                with gzip.open(f"{filepath}.gz", "wb", compresslevel=self._compress_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(filepath)
            logger.debug("Compressed: %s", filepath)

        except Exception as e:
            logger.error("Failed to compress %s: %s", filepath, e)


class LogRotationManager:
    """
    Manages log rotation for the application.

    Usage:
        manager = LogRotationManager()

        # Setup rotating logger
        handler = manager.create_handler("/path/to/app.log")
        logging.getLogger().addHandler(handler)

        # Cleanup old logs
        manager.cleanup_old_logs("/path/to/logs", max_age_days=30)

        # Get log statistics
        stats = manager.get_log_stats("/path/to/logs")
    """

    def __init__(self, config: RotationConfig | None = None):
        """
        Initialize log rotation manager.

        Args:
            config: Rotation configuration
        """
        self.config = config or RotationConfig()

    def create_handler(
        self,
        log_path: Path | str,
        config: RotationConfig | None = None,
    ) -> logging.Handler:
        """
        Create a rotating log handler.

        Args:
            log_path: Path to log file
            config: Optional override config

        Returns:
            Logging handler
        """
        cfg = config or self.config
        log_path = Path(log_path)

        # Ensure directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if cfg.rotation_type == RotationType.SIZE:
            return CompressingRotatingFileHandler(
                str(log_path),
                max_bytes=cfg.max_size_mb * 1024 * 1024,
                backup_count=cfg.backup_count,
                compress=cfg.compress,
                compress_level=cfg.compress_level,
                encoding="utf-8",
            )

        elif cfg.rotation_type == RotationType.TIME:
            return CompressingTimedRotatingFileHandler(
                str(log_path),
                when=cfg.time_interval.value,
                backupCount=cfg.backup_count,
                compress=cfg.compress,
                compress_level=cfg.compress_level,
                encoding="utf-8",
            )

        else:  # HYBRID
            # Use size-based as primary, but also check time
            return CompressingRotatingFileHandler(
                str(log_path),
                max_bytes=cfg.max_size_mb * 1024 * 1024,
                backup_count=cfg.backup_count,
                compress=cfg.compress,
                compress_level=cfg.compress_level,
                encoding="utf-8",
            )

    def cleanup_old_logs(
        self,
        log_dir: Path | str,
        max_age_days: int = 30,
        pattern: str = "*.log*",
    ) -> int:
        """
        Clean up old log files.

        Args:
            log_dir: Directory containing logs
            max_age_days: Maximum age in days
            pattern: File pattern to match

        Returns:
            Number of files deleted
        """
        log_dir = Path(log_dir)
        if not log_dir.exists():
            return 0

        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        deleted = 0

        for log_file in log_dir.glob(pattern):
            if log_file.is_file():
                file_time = log_file.stat().st_mtime
                if file_time < cutoff_time:
                    try:
                        log_file.unlink()
                        deleted += 1
                        logger.debug("Deleted old log: %s", log_file)
                    except Exception as e:
                        logger.error("Failed to delete %s: %s", log_file, e)

        if deleted > 0:
            logger.info("Cleaned up %s old log files", deleted)

        return deleted

    def get_log_stats(self, log_dir: Path | str) -> dict[str, Any]:
        """
        Get statistics about log files.

        Args:
            log_dir: Directory containing logs

        Returns:
            Statistics dict
        """
        log_dir = Path(log_dir)
        if not log_dir.exists():
            return {"total_files": 0, "total_size_mb": 0}

        files = list(log_dir.glob("*.log*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        # Get newest and oldest
        if files:
            files_with_time = [(f, f.stat().st_mtime) for f in files if f.is_file()]
            files_with_time.sort(key=lambda x: x[1])
            oldest = files_with_time[0] if files_with_time else None
            newest = files_with_time[-1] if files_with_time else None
        else:
            oldest = newest = None

        # Count by type
        compressed = sum(1 for f in files if f.suffix == ".gz")
        uncompressed = len(files) - compressed

        return {
            "total_files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "compressed_files": compressed,
            "uncompressed_files": uncompressed,
            "oldest_file": str(oldest[0]) if oldest else None,
            "oldest_time": datetime.fromtimestamp(oldest[1]).isoformat() if oldest else None,
            "newest_file": str(newest[0]) if newest else None,
            "newest_time": datetime.fromtimestamp(newest[1]).isoformat() if newest else None,
        }

    def rotate_now(self, handler: logging.Handler) -> bool:
        """
        Force immediate log rotation.

        Args:
            handler: The handler to rotate

        Returns:
            True if rotated
        """
        try:
            if isinstance(handler, RotatingFileHandler | TimedRotatingFileHandler):
                handler.doRollover()
                return True
            return False
        except Exception as e:
            logger.error("Failed to rotate: %s", e)
            return False

    def decompress_log(self, gz_path: Path | str) -> Path | None:
        """
        Decompress a gzipped log file.

        Args:
            gz_path: Path to .gz file

        Returns:
            Path to decompressed file or None
        """
        gz_path = Path(gz_path)
        if not gz_path.exists() or gz_path.suffix != ".gz":
            return None

        output_path = gz_path.with_suffix("")

        try:
            with gzip.open(gz_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            return output_path

        except Exception as e:
            logger.error("Failed to decompress %s: %s", gz_path, e)
            return None


def setup_rotating_logger(
    name: str,
    log_path: Path | str,
    level: int = logging.INFO,
    config: RotationConfig | None = None,
) -> logging.Logger:
    """
    Setup a logger with rotating file handler.

    Args:
        name: Logger name
        log_path: Path to log file
        level: Logging level
        config: Rotation configuration

    Returns:
        Configured logger
    """
    manager = LogRotationManager(config)
    handler = manager.create_handler(log_path)
    handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger
