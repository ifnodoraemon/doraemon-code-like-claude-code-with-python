"""
Internationalization (i18n) Support

Multi-language interface support.

Features:
- Multiple language support
- Dynamic language switching
- Message formatting with variables
- Locale-aware formatting
- Translation loading from files
"""

import json
import locale
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Built-in translations
BUILTIN_TRANSLATIONS = {
    "en": {
        # General
        "welcome": "Welcome to the agent!",
        "goodbye": "Goodbye!",
        "yes": "Yes",
        "no": "No",
        "ok": "OK",
        "cancel": "Cancel",
        "error": "Error",
        "warning": "Warning",
        "success": "Success",
        "info": "Info",
        # Commands
        "cmd.help": "Show help",
        "cmd.exit": "Exit",
        "cmd.clear": "Clear conversation",
        "cmd.mode": "Switch mode",
        "cmd.model": "Switch model",
        "cmd.cost": "Show cost statistics",
        "cmd.history": "Show command history",
        "cmd.sessions": "List sessions",
        "cmd.resume": "Resume session",
        "cmd.checkpoints": "List checkpoints",
        "cmd.rewind": "Rewind to checkpoint",
        "cmd.tasks": "List background tasks",
        "cmd.doctor": "Run health checks",
        "cmd.theme": "Change theme",
        # Modes
        "mode.plan": "Planning mode (read-only)",
        "mode.build": "Build mode",
        "mode.switched": "Switched to {mode} mode",
        # Prompts
        "prompt.input": "You ({mode})",
        "prompt.confirm": "Are you sure? [y/N]",
        "prompt.approve": "Approve this operation? [y/N]",
        # Status
        "status.loading": "Loading...",
        "status.processing": "Processing...",
        "status.complete": "Complete",
        "status.failed": "Failed",
        "status.cancelled": "Cancelled",
        # Errors
        "error.unknown_command": "Unknown command: {command}",
        "error.file_not_found": "File not found: {path}",
        "error.permission_denied": "Permission denied: {operation}",
        "error.api_error": "API error: {message}",
        "error.network": "Network error: {message}",
        "error.timeout": "Operation timed out",
        # Tasks
        "task.started": "Task started: {name}",
        "task.completed": "Task completed: {name}",
        "task.failed": "Task failed: {name}",
        "task.interrupted": "Task interrupted: {name}",
        "task.resumed": "Task resumed: {name}",
        "task.recovery_prompt": "Found {count} interrupted tasks. Resume? [y/N]",
        # Notifications
        "notify.build_complete": "Build completed successfully",
        "notify.build_failed": "Build failed",
        "notify.task_complete": "Task completed",
        "notify.error": "An error occurred",
        # Cost
        "cost.session": "Session cost: ${cost}",
        "cost.daily": "Daily cost: ${cost}",
        "cost.budget_warning": "Warning: Approaching budget limit",
        "cost.budget_exceeded": "Budget limit exceeded",
        # Checkpoints
        "checkpoint.created": "Checkpoint created: {id}",
        "checkpoint.restored": "Restored to checkpoint: {id}",
        "checkpoint.list_empty": "No checkpoints available",
        # Doctor
        "doctor.running": "Running health checks...",
        "doctor.complete": "Health check complete",
        "doctor.issues_found": "Found {count} issues",
    },
    "zh": {
        # 通用
        "welcome": "欢迎使用智能体！",
        "goodbye": "再见！",
        "yes": "是",
        "no": "否",
        "ok": "确定",
        "cancel": "取消",
        "error": "错误",
        "warning": "警告",
        "success": "成功",
        "info": "信息",
        # 命令
        "cmd.help": "显示帮助",
        "cmd.exit": "退出",
        "cmd.clear": "清除对话",
        "cmd.mode": "切换模式",
        "cmd.model": "切换模型",
        "cmd.cost": "显示费用统计",
        "cmd.history": "显示命令历史",
        "cmd.sessions": "列出会话",
        "cmd.resume": "恢复会话",
        "cmd.checkpoints": "列出检查点",
        "cmd.rewind": "回滚到检查点",
        "cmd.tasks": "列出后台任务",
        "cmd.doctor": "运行健康检查",
        "cmd.theme": "更改主题",
        # 模式
        "mode.plan": "规划模式（只读）",
        "mode.build": "构建模式",
        "mode.switched": "已切换到 {mode} 模式",
        # 提示
        "prompt.input": "你 ({mode})",
        "prompt.confirm": "确定吗？[y/N]",
        "prompt.approve": "批准此操作？[y/N]",
        # 状态
        "status.loading": "加载中...",
        "status.processing": "处理中...",
        "status.complete": "完成",
        "status.failed": "失败",
        "status.cancelled": "已取消",
        # 错误
        "error.unknown_command": "未知命令：{command}",
        "error.file_not_found": "文件未找到：{path}",
        "error.permission_denied": "权限被拒绝：{operation}",
        "error.api_error": "API 错误：{message}",
        "error.network": "网络错误：{message}",
        "error.timeout": "操作超时",
        # 任务
        "task.started": "任务已启动：{name}",
        "task.completed": "任务完成：{name}",
        "task.failed": "任务失败：{name}",
        "task.interrupted": "任务中断：{name}",
        "task.resumed": "任务已恢复：{name}",
        "task.recovery_prompt": "发现 {count} 个中断的任务。是否恢复？[y/N]",
        # 通知
        "notify.build_complete": "构建成功完成",
        "notify.build_failed": "构建失败",
        "notify.task_complete": "任务完成",
        "notify.error": "发生错误",
        # 费用
        "cost.session": "会话费用：${cost}",
        "cost.daily": "每日费用：${cost}",
        "cost.budget_warning": "警告：即将达到预算限制",
        "cost.budget_exceeded": "已超出预算限制",
        # 检查点
        "checkpoint.created": "检查点已创建：{id}",
        "checkpoint.restored": "已恢复到检查点：{id}",
        "checkpoint.list_empty": "没有可用的检查点",
        # 健康检查
        "doctor.running": "正在运行健康检查...",
        "doctor.complete": "健康检查完成",
        "doctor.issues_found": "发现 {count} 个问题",
    },
    "ja": {
        # 一般
        "welcome": "エージェントへようこそ！",
        "goodbye": "さようなら！",
        "yes": "はい",
        "no": "いいえ",
        "ok": "OK",
        "cancel": "キャンセル",
        "error": "エラー",
        "warning": "警告",
        "success": "成功",
        "info": "情報",
        # コマンド
        "cmd.help": "ヘルプを表示",
        "cmd.exit": "終了",
        "cmd.clear": "会話をクリア",
        "cmd.mode": "モードを切り替え",
        # ステータス
        "status.loading": "読み込み中...",
        "status.processing": "処理中...",
        "status.complete": "完了",
        "status.failed": "失敗",
    },
}


@dataclass
class I18nConfig:
    """Internationalization configuration."""

    default_locale: str = "en"
    fallback_locale: str = "en"
    translations_dir: Path | None = None
    auto_detect: bool = True


class I18n:
    """
    Internationalization manager.

    Usage:
        i18n = I18n()

        # Get translation
        msg = i18n.t("welcome")
        msg = i18n.t("error.file_not_found", path="/some/path")

        # Change language
        i18n.set_locale("zh")

        # Get current locale
        locale = i18n.get_locale()
    """

    def __init__(self, config: I18nConfig | None = None):
        """
        Initialize i18n manager.

        Args:
            config: I18n configuration
        """
        self.config = config or I18nConfig()
        self._translations: dict[str, dict[str, str]] = {}
        self._current_locale = self.config.default_locale

        # Load built-in translations
        self._translations.update(BUILTIN_TRANSLATIONS)

        # Load custom translations
        if self.config.translations_dir:
            self._load_translations(self.config.translations_dir)

        # Auto-detect locale
        if self.config.auto_detect:
            self._detect_locale()

    def _detect_locale(self):
        """Detect system locale."""
        # Check environment variable
        lang = os.getenv("AGENT_LANG") or os.getenv("LANG") or os.getenv("LANGUAGE")

        if lang:
            # Extract language code (e.g., "en_US.UTF-8" -> "en")
            lang_code = lang.split("_")[0].split(".")[0]
            if lang_code in self._translations:
                self._current_locale = lang_code
                return

        # Try system locale
        try:
            sys_locale = locale.getdefaultlocale()[0]
            if sys_locale:
                lang_code = sys_locale.split("_")[0]
                if lang_code in self._translations:
                    self._current_locale = lang_code
        except Exception:
            pass

    def _load_translations(self, directory: Path):
        """Load translations from directory."""
        if not directory.exists():
            return

        for file in directory.glob("*.json"):
            try:
                lang_code = file.stem
                data = json.loads(file.read_text(encoding="utf-8"))

                if lang_code in self._translations:
                    self._translations[lang_code].update(data)
                else:
                    self._translations[lang_code] = data

                logger.debug("Loaded translations for: %s", lang_code)

            except Exception as e:
                logger.warning("Failed to load translation file %s: %s", file, e)

    def t(self, key: str, **kwargs) -> str:
        """
        Get translation for a key.

        Args:
            key: Translation key (e.g., "error.file_not_found")
            **kwargs: Variables for string formatting

        Returns:
            Translated string
        """
        # Try current locale
        text = self._get_translation(self._current_locale, key)

        # Fall back to fallback locale
        if text is None and self._current_locale != self.config.fallback_locale:
            text = self._get_translation(self.config.fallback_locale, key)

        # Fall back to key itself
        if text is None:
            text = key

        # Format with variables
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass

        return text

    def _get_translation(self, locale_code: str, key: str) -> str | None:
        """Get translation from a specific locale."""
        if locale_code not in self._translations:
            return None

        translations = self._translations[locale_code]
        return translations.get(key)

    def set_locale(self, locale_code: str) -> bool:
        """
        Set current locale.

        Args:
            locale_code: Locale code (e.g., "en", "zh", "ja")

        Returns:
            True if locale was set
        """
        if locale_code not in self._translations:
            logger.warning("Unknown locale: %s", locale_code)
            return False

        self._current_locale = locale_code
        logger.info("Locale set to: %s", locale_code)
        return True

    def get_locale(self) -> str:
        """Get current locale."""
        return self._current_locale

    def get_available_locales(self) -> list[str]:
        """Get list of available locales."""
        return list(self._translations.keys())

    def add_translations(self, locale_code: str, translations: dict[str, str]):
        """
        Add translations for a locale.

        Args:
            locale_code: Locale code
            translations: Translation dict
        """
        if locale_code not in self._translations:
            self._translations[locale_code] = {}
        self._translations[locale_code].update(translations)

    def has_key(self, key: str) -> bool:
        """Check if translation key exists."""
        return self._get_translation(self._current_locale, key) is not None

    def format_number(self, number: float, decimal_places: int = 2) -> str:
        """Format number according to locale."""
        try:
            # Set locale for formatting
            if self._current_locale == "zh":
                return f"{number:,.{decimal_places}f}"
            elif self._current_locale == "ja":
                return f"{number:,.{decimal_places}f}"
            else:
                return f"{number:,.{decimal_places}f}"
        except Exception:
            return str(round(number, decimal_places))

    def format_date(self, timestamp: float) -> str:
        """Format timestamp according to locale."""
        import datetime

        dt = datetime.datetime.fromtimestamp(timestamp)

        if self._current_locale == "zh":
            return dt.strftime("%Y年%m月%d日 %H:%M")
        elif self._current_locale == "ja":
            return dt.strftime("%Y年%m月%d日 %H:%M")
        else:
            return dt.strftime("%Y-%m-%d %H:%M")

    def get_summary(self) -> dict[str, Any]:
        """Get i18n summary."""
        return {
            "current_locale": self._current_locale,
            "available_locales": self.get_available_locales(),
            "total_keys": sum(len(t) for t in self._translations.values()),
        }


# Global instance
_i18n: I18n | None = None


def get_i18n() -> I18n:
    """Get the global i18n instance."""
    global _i18n
    if _i18n is None:
        _i18n = I18n()
    return _i18n


# Convenience function
def t(key: str, **kwargs) -> str:
    """Get translation for a key."""
    return get_i18n().t(key, **kwargs)
