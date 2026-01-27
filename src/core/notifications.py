"""
Notification System

Desktop and broadcast notifications.

Features:
- Desktop notifications (cross-platform)
- Sound alerts
- Notification history
- Do-not-disturb mode
- Notification channels
"""

import logging
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification importance level."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Notification delivery channels."""

    DESKTOP = "desktop"  # System notification
    CONSOLE = "console"  # Console output
    SOUND = "sound"  # Sound alert
    LOG = "log"  # Log only
    WEBHOOK = "webhook"  # HTTP webhook


@dataclass
class Notification:
    """A notification message."""

    title: str
    message: str
    level: NotificationLevel = NotificationLevel.INFO
    channels: list[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.DESKTOP, NotificationChannel.CONSOLE]
    )
    icon: str | None = None
    sound: str | None = None
    action_url: str | None = None
    timeout: int = 5  # Seconds to display
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "message": self.message,
            "level": self.level.value,
            "channels": [c.value for c in self.channels],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class NotificationConfig:
    """Notification system configuration."""

    enabled: bool = True
    do_not_disturb: bool = False
    desktop_enabled: bool = True
    sound_enabled: bool = False
    default_timeout: int = 5
    max_history: int = 100
    webhook_url: str | None = None


class DesktopNotifier:
    """
    Cross-platform desktop notification sender.

    Supports:
    - macOS (osascript/terminal-notifier)
    - Linux (notify-send/libnotify)
    - Windows (toast notifications)
    """

    def __init__(self):
        self._system = platform.system()
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if desktop notifications are available."""
        if self._system == "Darwin":
            # macOS always has osascript
            return True
        elif self._system == "Linux":
            # Check for notify-send
            try:
                subprocess.run(
                    ["which", "notify-send"],
                    capture_output=True,
                    check=True,
                )
                return True
            except subprocess.CalledProcessError:
                return False
        elif self._system == "Windows":
            try:
                # Check for Windows 10+ toast notifications
                return True
            except Exception:
                return False
        return False

    def is_available(self) -> bool:
        """Check if notifications are available."""
        return self._available

    def send(self, notification: Notification) -> bool:
        """
        Send a desktop notification.

        Args:
            notification: Notification to send

        Returns:
            True if sent successfully
        """
        if not self._available:
            logger.warning("Desktop notifications not available")
            return False

        try:
            if self._system == "Darwin":
                return self._send_macos(notification)
            elif self._system == "Linux":
                return self._send_linux(notification)
            elif self._system == "Windows":
                return self._send_windows(notification)
            return False
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    def _send_macos(self, notification: Notification) -> bool:
        """Send notification on macOS."""
        script = f'''
        display notification "{notification.message}" with title "{notification.title}"
        '''

        if notification.sound:
            script += f' sound name "{notification.sound}"'

        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
        )
        return True

    def _send_linux(self, notification: Notification) -> bool:
        """Send notification on Linux."""
        cmd = [
            "notify-send",
            notification.title,
            notification.message,
            f"--expire-time={notification.timeout * 1000}",
        ]

        # Map level to urgency
        urgency_map = {
            NotificationLevel.INFO: "low",
            NotificationLevel.SUCCESS: "normal",
            NotificationLevel.WARNING: "normal",
            NotificationLevel.ERROR: "critical",
            NotificationLevel.CRITICAL: "critical",
        }
        cmd.extend(["--urgency", urgency_map.get(notification.level, "normal")])

        if notification.icon:
            cmd.extend(["--icon", notification.icon])

        subprocess.run(cmd, capture_output=True)
        return True

    def _send_windows(self, notification: Notification) -> bool:
        """Send notification on Windows."""
        try:
            from win10toast import ToastNotifier

            toaster = ToastNotifier()
            toaster.show_toast(
                notification.title,
                notification.message,
                duration=notification.timeout,
                threaded=True,
            )
            return True
        except ImportError:
            # Fallback to PowerShell
            script = f'''
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
            $template = "<toast><visual><binding template='ToastText02'><text id='1'>{notification.title}</text><text id='2'>{notification.message}</text></binding></visual></toast>"
            $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
            $xml.LoadXml($template)
            $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Doraemon").Show($toast)
            '''

            subprocess.run(
                ["powershell", "-Command", script],
                capture_output=True,
            )
            return True


class NotificationManager:
    """
    Manages notifications across all channels.

    Usage:
        notifier = NotificationManager()

        # Send notification
        notifier.notify(
            title="Task Complete",
            message="Build finished successfully",
            level=NotificationLevel.SUCCESS
        )

        # With specific channels
        notifier.notify(
            title="Error",
            message="Build failed",
            channels=[NotificationChannel.DESKTOP, NotificationChannel.SOUND]
        )

        # Quick helpers
        notifier.info("Starting build...")
        notifier.success("Build complete!")
        notifier.error("Build failed!")

        # Do not disturb
        notifier.set_dnd(True)
    """

    def __init__(self, config: NotificationConfig | None = None):
        """
        Initialize notification manager.

        Args:
            config: Notification configuration
        """
        self.config = config or NotificationConfig()
        self._desktop = DesktopNotifier()
        self._history: list[Notification] = []
        self._callbacks: list[Callable[[Notification], None]] = []

    def notify(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: list[NotificationChannel] | None = None,
        **kwargs,
    ) -> bool:
        """
        Send a notification.

        Args:
            title: Notification title
            message: Notification message
            level: Notification level
            channels: Delivery channels
            **kwargs: Additional notification options

        Returns:
            True if at least one channel succeeded
        """
        if not self.config.enabled:
            return False

        # Create notification
        notification = Notification(
            title=title,
            message=message,
            level=level,
            channels=channels or [NotificationChannel.DESKTOP, NotificationChannel.CONSOLE],
            timeout=kwargs.get("timeout", self.config.default_timeout),
            icon=kwargs.get("icon"),
            sound=kwargs.get("sound"),
            action_url=kwargs.get("action_url"),
            metadata=kwargs.get("metadata", {}),
        )

        # Add to history
        self._history.append(notification)
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history:]

        # Check DND
        if self.config.do_not_disturb and level != NotificationLevel.CRITICAL:
            # Still log, but don't alert
            self._send_log(notification)
            return True

        # Send to channels
        success = False
        for channel in notification.channels:
            if self._send_channel(channel, notification):
                success = True

        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

        return success

    def _send_channel(
        self, channel: NotificationChannel, notification: Notification
    ) -> bool:
        """Send to a specific channel."""
        if channel == NotificationChannel.DESKTOP:
            if self.config.desktop_enabled:
                return self._desktop.send(notification)
        elif channel == NotificationChannel.CONSOLE:
            return self._send_console(notification)
        elif channel == NotificationChannel.SOUND:
            if self.config.sound_enabled:
                return self._send_sound(notification)
        elif channel == NotificationChannel.LOG:
            return self._send_log(notification)
        elif channel == NotificationChannel.WEBHOOK:
            return self._send_webhook(notification)
        return False

    def _send_console(self, notification: Notification) -> bool:
        """Send to console."""
        level_icons = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.SUCCESS: "✅",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨",
        }

        icon = level_icons.get(notification.level, "•")
        print(f"\n{icon} {notification.title}: {notification.message}")
        return True

    def _send_sound(self, notification: Notification) -> bool:
        """Play notification sound."""
        try:
            system = platform.system()

            if system == "Darwin":
                sound = notification.sound or "Glass"
                subprocess.run(
                    ["afplay", f"/System/Library/Sounds/{sound}.aiff"],
                    capture_output=True,
                )
            elif system == "Linux":
                subprocess.run(
                    ["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
                    capture_output=True,
                )
            elif system == "Windows":
                import winsound
                winsound.MessageBeep()

            return True
        except Exception as e:
            logger.debug(f"Sound failed: {e}")
            return False

    def _send_log(self, notification: Notification) -> bool:
        """Log notification."""
        level_map = {
            NotificationLevel.INFO: logging.INFO,
            NotificationLevel.SUCCESS: logging.INFO,
            NotificationLevel.WARNING: logging.WARNING,
            NotificationLevel.ERROR: logging.ERROR,
            NotificationLevel.CRITICAL: logging.CRITICAL,
        }

        log_level = level_map.get(notification.level, logging.INFO)
        logger.log(log_level, f"{notification.title}: {notification.message}")
        return True

    def _send_webhook(self, notification: Notification) -> bool:
        """Send to webhook."""
        if not self.config.webhook_url:
            return False

        try:
            import urllib.request

            data = notification.to_dict()
            req = urllib.request.Request(
                self.config.webhook_url,
                data=str(data).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception as e:
            logger.error(f"Webhook failed: {e}")
            return False

    # Convenience methods
    def info(self, message: str, title: str = "Info"):
        """Send info notification."""
        return self.notify(title, message, NotificationLevel.INFO)

    def success(self, message: str, title: str = "Success"):
        """Send success notification."""
        return self.notify(title, message, NotificationLevel.SUCCESS)

    def warning(self, message: str, title: str = "Warning"):
        """Send warning notification."""
        return self.notify(title, message, NotificationLevel.WARNING)

    def error(self, message: str, title: str = "Error"):
        """Send error notification."""
        return self.notify(title, message, NotificationLevel.ERROR)

    def critical(self, message: str, title: str = "Critical"):
        """Send critical notification."""
        return self.notify(
            title, message, NotificationLevel.CRITICAL,
            channels=[NotificationChannel.DESKTOP, NotificationChannel.CONSOLE, NotificationChannel.SOUND]
        )

    def on_notification(self, callback: Callable[[Notification], None]):
        """Register notification callback."""
        self._callbacks.append(callback)

    def set_dnd(self, enabled: bool):
        """Set do-not-disturb mode."""
        self.config.do_not_disturb = enabled

    def get_history(self, limit: int = 50) -> list[Notification]:
        """Get notification history."""
        return self._history[-limit:]

    def clear_history(self):
        """Clear notification history."""
        self._history.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get notification system summary."""
        return {
            "enabled": self.config.enabled,
            "dnd": self.config.do_not_disturb,
            "desktop_available": self._desktop.is_available(),
            "history_count": len(self._history),
        }


# Global instance
_notification_manager: NotificationManager | None = None


def get_notification_manager() -> NotificationManager:
    """Get the global notification manager."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


# Convenience functions
def notify(title: str, message: str, level: NotificationLevel = NotificationLevel.INFO):
    """Send a notification."""
    return get_notification_manager().notify(title, message, level)


def notify_success(message: str, title: str = "Success"):
    """Send success notification."""
    return get_notification_manager().success(message, title)


def notify_error(message: str, title: str = "Error"):
    """Send error notification."""
    return get_notification_manager().error(message, title)
