"""Comprehensive tests for notifications.py"""

import subprocess
import time
from unittest.mock import MagicMock, Mock, patch

from src.core.notifications import (
    DesktopNotifier,
    Notification,
    NotificationChannel,
    NotificationConfig,
    NotificationLevel,
    NotificationManager,
    get_notification_manager,
    notify,
    notify_error,
    notify_success,
)


class TestNotificationLevel:
    """Tests for NotificationLevel enum."""

    def test_notification_level_values(self):
        """Test all notification level values."""
        assert NotificationLevel.INFO.value == "info"
        assert NotificationLevel.SUCCESS.value == "success"
        assert NotificationLevel.WARNING.value == "warning"
        assert NotificationLevel.ERROR.value == "error"
        assert NotificationLevel.CRITICAL.value == "critical"

    def test_notification_level_members(self):
        """Test all notification level members exist."""
        levels = [
            NotificationLevel.INFO,
            NotificationLevel.SUCCESS,
            NotificationLevel.WARNING,
            NotificationLevel.ERROR,
            NotificationLevel.CRITICAL,
        ]
        assert len(levels) == 5


class TestNotificationChannel:
    """Tests for NotificationChannel enum."""

    def test_notification_channel_values(self):
        """Test all notification channel values."""
        assert NotificationChannel.DESKTOP.value == "desktop"
        assert NotificationChannel.CONSOLE.value == "console"
        assert NotificationChannel.SOUND.value == "sound"
        assert NotificationChannel.LOG.value == "log"
        assert NotificationChannel.WEBHOOK.value == "webhook"

    def test_notification_channel_members(self):
        """Test all notification channel members exist."""
        channels = [
            NotificationChannel.DESKTOP,
            NotificationChannel.CONSOLE,
            NotificationChannel.SOUND,
            NotificationChannel.LOG,
            NotificationChannel.WEBHOOK,
        ]
        assert len(channels) == 5


class TestNotification:
    """Tests for Notification dataclass."""

    def test_notification_creation_minimal(self):
        """Test creating notification with minimal parameters."""
        notif = Notification(title="Test", message="Test message")
        assert notif.title == "Test"
        assert notif.message == "Test message"
        assert notif.level == NotificationLevel.INFO
        assert notif.icon is None
        assert notif.sound is None
        assert notif.action_url is None
        assert notif.timeout == 5
        assert notif.metadata == {}

    def test_notification_creation_full(self):
        """Test creating notification with all parameters."""
        metadata = {"key": "value"}
        notif = Notification(
            title="Full Test",
            message="Full message",
            level=NotificationLevel.ERROR,
            channels=[NotificationChannel.DESKTOP, NotificationChannel.SOUND],
            icon="error.png",
            sound="alert.wav",
            action_url="https://example.com",
            timeout=10,
            metadata=metadata,
        )
        assert notif.title == "Full Test"
        assert notif.message == "Full message"
        assert notif.level == NotificationLevel.ERROR
        assert notif.icon == "error.png"
        assert notif.sound == "alert.wav"
        assert notif.action_url == "https://example.com"
        assert notif.timeout == 10
        assert notif.metadata == metadata

    def test_notification_default_channels(self):
        """Test notification has default channels."""
        notif = Notification(title="Test", message="Test")
        assert NotificationChannel.DESKTOP in notif.channels
        assert NotificationChannel.CONSOLE in notif.channels
        assert len(notif.channels) == 2

    def test_notification_timestamp_auto_set(self):
        """Test notification timestamp is automatically set."""
        before = time.time()
        notif = Notification(title="Test", message="Test")
        after = time.time()
        assert before <= notif.timestamp <= after

    def test_notification_to_dict(self):
        """Test converting notification to dictionary."""
        notif = Notification(
            title="Test",
            message="Message",
            level=NotificationLevel.SUCCESS,
            channels=[NotificationChannel.CONSOLE],
            metadata={"key": "value"},
        )
        data = notif.to_dict()
        assert data["title"] == "Test"
        assert data["message"] == "Message"
        assert data["level"] == "success"
        assert data["channels"] == ["console"]
        assert data["metadata"] == {"key": "value"}
        assert "timestamp" in data

    def test_notification_to_dict_all_levels(self):
        """Test to_dict with all notification levels."""
        for level in NotificationLevel:
            notif = Notification(title="Test", message="Test", level=level)
            data = notif.to_dict()
            assert data["level"] == level.value

    def test_notification_to_dict_all_channels(self):
        """Test to_dict with all notification channels."""
        channels = list(NotificationChannel)
        notif = Notification(title="Test", message="Test", channels=channels)
        data = notif.to_dict()
        assert len(data["channels"]) == len(channels)
        for channel in channels:
            assert channel.value in data["channels"]


class TestNotificationConfig:
    """Tests for NotificationConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = NotificationConfig()
        assert config.enabled is True
        assert config.do_not_disturb is False
        assert config.desktop_enabled is True
        assert config.sound_enabled is False
        assert config.default_timeout == 5
        assert config.max_history == 100
        assert config.webhook_url is None

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = NotificationConfig(
            enabled=False,
            do_not_disturb=True,
            desktop_enabled=False,
            sound_enabled=True,
            default_timeout=10,
            max_history=200,
            webhook_url="https://webhook.example.com",
        )
        assert config.enabled is False
        assert config.do_not_disturb is True
        assert config.desktop_enabled is False
        assert config.sound_enabled is True
        assert config.default_timeout == 10
        assert config.max_history == 200
        assert config.webhook_url == "https://webhook.example.com"


class TestDesktopNotifier:
    """Tests for DesktopNotifier class."""

    def test_desktop_notifier_initialization(self):
        """Test DesktopNotifier initialization."""
        notifier = DesktopNotifier()
        assert notifier._system is not None
        assert isinstance(notifier._available, bool)

    def test_is_available(self):
        """Test is_available method."""
        notifier = DesktopNotifier()
        result = notifier.is_available()
        assert isinstance(result, bool)

    @patch("platform.system", return_value="Darwin")
    def test_check_availability_macos(self, mock_system):
        """Test availability check on macOS."""
        notifier = DesktopNotifier()
        assert notifier._available is True

    @patch("platform.system", return_value="Linux")
    @patch("subprocess.run")
    def test_check_availability_linux_available(self, mock_run, mock_system):
        """Test availability check on Linux when notify-send is available."""
        mock_run.return_value = MagicMock(returncode=0)
        notifier = DesktopNotifier()
        assert notifier._available is True

    @patch("platform.system", return_value="Linux")
    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "which"))
    def test_check_availability_linux_unavailable(self, mock_run, mock_system):
        """Test availability check on Linux when notify-send is unavailable."""
        notifier = DesktopNotifier()
        assert notifier._available is False

    @patch("platform.system", return_value="Windows")
    def test_check_availability_windows(self, mock_system):
        """Test availability check on Windows."""
        notifier = DesktopNotifier()
        assert notifier._available is True

    @patch("platform.system", return_value="Unknown")
    def test_check_availability_unknown_system(self, mock_system):
        """Test availability check on unknown system."""
        notifier = DesktopNotifier()
        assert notifier._available is False

    @patch.object(DesktopNotifier, "_send_macos")
    @patch.object(DesktopNotifier, "_check_availability", return_value=True)
    @patch("platform.system", return_value="Darwin")
    def test_send_macos(self, mock_system, mock_check, mock_send_macos):
        """Test sending notification on macOS."""
        mock_send_macos.return_value = True
        notifier = DesktopNotifier()
        notif = Notification(title="Test", message="Test message")
        result = notifier.send(notif)
        assert result is True

    @patch.object(DesktopNotifier, "_send_linux")
    @patch.object(DesktopNotifier, "_check_availability", return_value=True)
    @patch("platform.system", return_value="Linux")
    def test_send_linux(self, mock_system, mock_check, mock_send_linux):
        """Test sending notification on Linux."""
        mock_send_linux.return_value = True
        notifier = DesktopNotifier()
        notif = Notification(title="Test", message="Test message")
        result = notifier.send(notif)
        assert result is True

    @patch.object(DesktopNotifier, "_send_windows")
    @patch.object(DesktopNotifier, "_check_availability", return_value=True)
    @patch("platform.system", return_value="Windows")
    def test_send_windows(self, mock_system, mock_check, mock_send_windows):
        """Test sending notification on Windows."""
        mock_send_windows.return_value = True
        notifier = DesktopNotifier()
        notif = Notification(title="Test", message="Test message")
        result = notifier.send(notif)
        assert result is True

    @patch.object(DesktopNotifier, "_check_availability", return_value=False)
    def test_send_unavailable(self, mock_check):
        """Test sending when notifications are unavailable."""
        notifier = DesktopNotifier()
        notif = Notification(title="Test", message="Test message")
        result = notifier.send(notif)
        assert result is False

    @patch.object(DesktopNotifier, "_send_macos", side_effect=Exception("Error"))
    @patch.object(DesktopNotifier, "_check_availability", return_value=True)
    @patch("platform.system", return_value="Darwin")
    def test_send_exception_handling(self, mock_system, mock_check, mock_send):
        """Test exception handling during send."""
        notifier = DesktopNotifier()
        notif = Notification(title="Test", message="Test message")
        result = notifier.send(notif)
        assert result is False

    @patch("subprocess.run")
    @patch.object(DesktopNotifier, "_check_availability", return_value=True)
    @patch("platform.system", return_value="Linux")
    def test_send_linux_with_icon(self, mock_system, mock_check, mock_run):
        """Test sending Linux notification with icon."""
        notifier = DesktopNotifier()
        notif = Notification(
            title="Test",
            message="Test message",
            icon="info.png",
            level=NotificationLevel.WARNING,
        )
        notifier._send_linux(notif)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "--icon" in args
        assert "info.png" in args
        assert "--urgency" in args
        assert "normal" in args


class TestNotificationManager:
    """Tests for NotificationManager class."""

    def test_manager_initialization_default(self):
        """Test NotificationManager initialization with defaults."""
        manager = NotificationManager()
        assert manager.config.enabled is True
        assert manager.config.do_not_disturb is False
        assert len(manager._history) == 0
        assert len(manager._callbacks) == 0

    def test_manager_initialization_custom_config(self):
        """Test NotificationManager initialization with custom config."""
        config = NotificationConfig(enabled=False, max_history=50)
        manager = NotificationManager(config)
        assert manager.config.enabled is False
        assert manager.config.max_history == 50

    def test_notify_disabled(self):
        """Test notify when notifications are disabled."""
        config = NotificationConfig(enabled=False)
        manager = NotificationManager(config)
        result = manager.notify("Test", "Message")
        assert result is False
        assert len(manager._history) == 0

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_basic(self, mock_send):
        """Test basic notification."""
        manager = NotificationManager()
        result = manager.notify("Test", "Message")
        assert result is True
        assert len(manager._history) == 1
        assert manager._history[0].title == "Test"
        assert manager._history[0].message == "Message"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_with_level(self, mock_send):
        """Test notification with specific level."""
        manager = NotificationManager()
        manager.notify("Test", "Message", level=NotificationLevel.ERROR)
        assert manager._history[0].level == NotificationLevel.ERROR

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_with_channels(self, mock_send):
        """Test notification with specific channels."""
        manager = NotificationManager()
        channels = [NotificationChannel.CONSOLE, NotificationChannel.LOG]
        manager.notify("Test", "Message", channels=channels)
        assert manager._history[0].channels == channels

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_with_kwargs(self, mock_send):
        """Test notification with additional kwargs."""
        manager = NotificationManager()
        manager.notify(
            "Test",
            "Message",
            timeout=10,
            icon="test.png",
            sound="alert.wav",
            action_url="https://example.com",
            metadata={"key": "value"},
        )
        notif = manager._history[0]
        assert notif.timeout == 10
        assert notif.icon == "test.png"
        assert notif.sound == "alert.wav"
        assert notif.action_url == "https://example.com"
        assert notif.metadata == {"key": "value"}

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_history_limit(self, mock_send):
        """Test notification history respects max_history limit."""
        config = NotificationConfig(max_history=5)
        manager = NotificationManager(config)
        for i in range(10):
            manager.notify(f"Test {i}", f"Message {i}")
        assert len(manager._history) == 5
        assert manager._history[0].title == "Test 5"
        assert manager._history[-1].title == "Test 9"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_dnd_non_critical(self, mock_send):
        """Test DND mode blocks non-critical notifications."""
        config = NotificationConfig(do_not_disturb=True)
        manager = NotificationManager(config)
        result = manager.notify("Test", "Message", level=NotificationLevel.INFO)
        assert result is True
        assert len(manager._history) == 1
        mock_send.assert_not_called()

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_dnd_critical(self, mock_send):
        """Test DND mode allows critical notifications."""
        config = NotificationConfig(do_not_disturb=True)
        manager = NotificationManager(config)
        result = manager.notify("Test", "Message", level=NotificationLevel.CRITICAL)
        assert result is True
        assert len(manager._history) == 1

    def test_notify_callback_registration(self):
        """Test registering notification callbacks."""
        manager = NotificationManager()
        callback = Mock()
        manager.on_notification(callback)
        assert callback in manager._callbacks

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_callback_execution(self, mock_send):
        """Test callbacks are executed on notification."""
        manager = NotificationManager()
        callback = Mock()
        manager.on_notification(callback)
        manager.notify("Test", "Message")
        callback.assert_called_once()
        notif = callback.call_args[0][0]
        assert notif.title == "Test"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_callback_exception_handling(self, mock_send):
        """Test callback exceptions don't break notification flow."""
        manager = NotificationManager()
        callback = Mock(side_effect=Exception("Callback error"))
        manager.on_notification(callback)
        result = manager.notify("Test", "Message")
        assert result is True

    def test_set_dnd(self):
        """Test setting do-not-disturb mode."""
        manager = NotificationManager()
        assert manager.config.do_not_disturb is False
        manager.set_dnd(True)
        assert manager.config.do_not_disturb is True
        manager.set_dnd(False)
        assert manager.config.do_not_disturb is False

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_get_history(self, mock_send):
        """Test getting notification history."""
        manager = NotificationManager()
        for i in range(10):
            manager.notify(f"Test {i}", f"Message {i}")
        history = manager.get_history(limit=5)
        assert len(history) == 5
        assert history[0].title == "Test 5"
        assert history[-1].title == "Test 9"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_get_history_default_limit(self, mock_send):
        """Test getting history with default limit."""
        manager = NotificationManager()
        for i in range(100):
            manager.notify(f"Test {i}", f"Message {i}")
        history = manager.get_history()
        assert len(history) == 50

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_clear_history(self, mock_send):
        """Test clearing notification history."""
        manager = NotificationManager()
        manager.notify("Test", "Message")
        assert len(manager._history) == 1
        manager.clear_history()
        assert len(manager._history) == 0

    def test_get_summary(self):
        """Test getting notification system summary."""
        config = NotificationConfig(enabled=True, do_not_disturb=False)
        manager = NotificationManager(config)
        summary = manager.get_summary()
        assert "enabled" in summary
        assert "dnd" in summary
        assert "desktop_available" in summary
        assert "history_count" in summary
        assert summary["enabled"] is True
        assert summary["dnd"] is False
        assert summary["history_count"] == 0

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_convenience_method_info(self, mock_send):
        """Test info convenience method."""
        manager = NotificationManager()
        manager.info("Info message")
        assert manager._history[0].level == NotificationLevel.INFO
        assert manager._history[0].message == "Info message"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_convenience_method_success(self, mock_send):
        """Test success convenience method."""
        manager = NotificationManager()
        manager.success("Success message")
        assert manager._history[0].level == NotificationLevel.SUCCESS
        assert manager._history[0].message == "Success message"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_convenience_method_warning(self, mock_send):
        """Test warning convenience method."""
        manager = NotificationManager()
        manager.warning("Warning message")
        assert manager._history[0].level == NotificationLevel.WARNING
        assert manager._history[0].message == "Warning message"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_convenience_method_error(self, mock_send):
        """Test error convenience method."""
        manager = NotificationManager()
        manager.error("Error message")
        assert manager._history[0].level == NotificationLevel.ERROR
        assert manager._history[0].message == "Error message"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_convenience_method_critical(self, mock_send):
        """Test critical convenience method."""
        manager = NotificationManager()
        manager.critical("Critical message")
        assert manager._history[0].level == NotificationLevel.CRITICAL
        assert manager._history[0].message == "Critical message"
        assert NotificationChannel.SOUND in manager._history[0].channels

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_send_console_channel(self, mock_send, capsys):
        """Test sending to console channel."""
        manager = NotificationManager()
        manager.notify(
            "Test",
            "Message",
            channels=[NotificationChannel.CONSOLE],
        )
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "Message" in captured.out

    @patch("logging.Logger.log")
    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_send_log_channel(self, mock_send, mock_log):
        """Test sending to log channel."""
        manager = NotificationManager()
        manager.notify(
            "Test",
            "Message",
            level=NotificationLevel.ERROR,
            channels=[NotificationChannel.LOG],
        )
        mock_log.assert_called()

    @patch("urllib.request.urlopen")
    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_send_webhook_channel(self, mock_send, mock_urlopen):
        """Test sending to webhook channel."""
        config = NotificationConfig(webhook_url="https://webhook.example.com")
        manager = NotificationManager(config)
        manager.notify(
            "Test",
            "Message",
            channels=[NotificationChannel.WEBHOOK],
        )
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen", side_effect=Exception("Network error"))
    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_send_webhook_error(self, mock_send, mock_urlopen):
        """Test webhook error handling."""
        config = NotificationConfig(webhook_url="https://webhook.example.com")
        manager = NotificationManager(config)
        result = manager.notify(
            "Test",
            "Message",
            channels=[NotificationChannel.WEBHOOK],
        )
        assert result is False

    @patch.object(DesktopNotifier, "send", return_value=False)
    def test_notify_all_channels_fail(self, mock_send):
        """Test when all channels fail."""
        manager = NotificationManager()
        result = manager.notify(
            "Test",
            "Message",
            channels=[NotificationChannel.DESKTOP],
        )
        assert result is False

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_multiple_channels(self, mock_send):
        """Test notification with multiple channels."""
        manager = NotificationManager()
        channels = [
            NotificationChannel.DESKTOP,
            NotificationChannel.CONSOLE,
            NotificationChannel.LOG,
        ]
        result = manager.notify("Test", "Message", channels=channels)
        assert result is True

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notify_filtering_by_priority(self, mock_send):
        """Test notification filtering based on priority/level."""
        manager = NotificationManager()
        levels = [
            NotificationLevel.INFO,
            NotificationLevel.WARNING,
            NotificationLevel.ERROR,
            NotificationLevel.CRITICAL,
        ]
        for level in levels:
            manager.notify("Test", "Message", level=level)
        assert len(manager._history) == 4
        assert manager._history[0].level == NotificationLevel.INFO
        assert manager._history[-1].level == NotificationLevel.CRITICAL


class TestGlobalNotificationFunctions:
    """Tests for global notification functions."""

    def test_get_notification_manager_singleton(self):
        """Test get_notification_manager returns singleton."""
        manager1 = get_notification_manager()
        manager2 = get_notification_manager()
        assert manager1 is manager2

    @patch.object(NotificationManager, "notify", return_value=True)
    def test_notify_function(self, mock_notify):
        """Test global notify function."""
        notify("Test", "Message", NotificationLevel.INFO)
        mock_notify.assert_called_once()

    @patch.object(NotificationManager, "success", return_value=True)
    def test_notify_success_function(self, mock_success):
        """Test global notify_success function."""
        notify_success("Success message")
        mock_success.assert_called_once()

    @patch.object(NotificationManager, "error", return_value=True)
    def test_notify_error_function(self, mock_error):
        """Test global notify_error function."""
        notify_error("Error message")
        mock_error.assert_called_once()


class TestNotificationIntegration:
    """Integration tests for notification system."""

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_full_notification_flow(self, mock_send):
        """Test complete notification flow."""
        config = NotificationConfig(max_history=10)
        manager = NotificationManager(config)
        callback_results = []

        def callback(notif):
            callback_results.append(notif)

        manager.on_notification(callback)
        manager.notify("Test", "Message", level=NotificationLevel.SUCCESS)
        assert len(manager._history) == 1
        assert len(callback_results) == 1
        assert callback_results[0].title == "Test"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notification_with_dnd_and_critical(self, mock_send):
        """Test DND mode with critical notification."""
        config = NotificationConfig(do_not_disturb=True)
        manager = NotificationManager(config)
        manager.notify("Info", "Info message", level=NotificationLevel.INFO)
        manager.notify("Critical", "Critical message", level=NotificationLevel.CRITICAL)
        assert len(manager._history) == 2
        assert manager._history[0].title == "Info"
        assert manager._history[1].title == "Critical"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notification_history_ordering(self, mock_send):
        """Test notification history maintains order."""
        manager = NotificationManager()
        for i in range(5):
            manager.notify(f"Test {i}", f"Message {i}")
        history = manager.get_history()
        for i, notif in enumerate(history):
            assert notif.title == f"Test {i}"

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_multiple_callbacks(self, mock_send):
        """Test multiple callbacks are all executed."""
        manager = NotificationManager()
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()
        manager.on_notification(callback1)
        manager.on_notification(callback2)
        manager.on_notification(callback3)
        manager.notify("Test", "Message")
        callback1.assert_called_once()
        callback2.assert_called_once()
        callback3.assert_called_once()

    @patch.object(DesktopNotifier, "send", return_value=True)
    def test_notification_metadata_preservation(self, mock_send):
        """Test metadata is preserved through notification flow."""
        manager = NotificationManager()
        metadata = {"user_id": "123", "action": "build_complete"}
        manager.notify("Test", "Message", metadata=metadata)
        notif = manager._history[0]
        assert notif.metadata == metadata
        data = notif.to_dict()
        assert data["metadata"] == metadata
