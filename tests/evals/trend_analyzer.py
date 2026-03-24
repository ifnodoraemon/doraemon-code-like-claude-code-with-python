"""
历史趋势分析系统

跟踪和分析模型性能变化，检测回归，生成趋势报告。
"""

import json
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class EvaluationRecord:
    """评估记录"""

    id: str
    version: str
    timestamp: datetime
    success_rate: float
    total_tasks: int
    avg_latency: float
    pass_at_1: float = 0.0
    pass_at_3: float = 0.0
    by_difficulty: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "success_rate": self.success_rate,
            "total_tasks": self.total_tasks,
            "avg_latency": self.avg_latency,
            "pass_at_1": self.pass_at_1,
            "pass_at_3": self.pass_at_3,
            "by_difficulty": self.by_difficulty,
            "by_category": self.by_category,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationRecord":
        return cls(
            id=data["id"],
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            success_rate=data["success_rate"],
            total_tasks=data["total_tasks"],
            avg_latency=data["avg_latency"],
            pass_at_1=data.get("pass_at_1", 0.0),
            pass_at_3=data.get("pass_at_3", 0.0),
            by_difficulty=data.get("by_difficulty", {}),
            by_category=data.get("by_category", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TrendResult:
    """趋势分析结果"""

    metric: str
    direction: str  # "up", "down", "stable"
    change_percent: float
    current_value: float
    previous_value: float
    data_points: int
    confidence: float  # 0-1


@dataclass
class RegressionResult:
    """回归检测结果"""

    has_regression: bool
    metric: str
    baseline_value: float
    current_value: float
    change_percent: float
    threshold: float
    message: str


class EvaluationStore:
    """
    评估结果持久化存储

    支持 SQLite 和 JSON 两种存储方式
    """

    def __init__(self, db_path: str = "eval_results/history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                success_rate REAL NOT NULL,
                total_tasks INTEGER NOT NULL,
                avg_latency REAL NOT NULL,
                pass_at_1 REAL DEFAULT 0,
                pass_at_3 REAL DEFAULT 0,
                by_difficulty TEXT,
                by_category TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_version ON evaluations(version)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluations(timestamp)
        """)

        conn.commit()
        conn.close()

    def save_evaluation(self, record: EvaluationRecord):
        """保存评估记录"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO evaluations
            (id, version, timestamp, success_rate, total_tasks, avg_latency,
             pass_at_1, pass_at_3, by_difficulty, by_category, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                record.id,
                record.version,
                record.timestamp.isoformat(),
                record.success_rate,
                record.total_tasks,
                record.avg_latency,
                record.pass_at_1,
                record.pass_at_3,
                json.dumps(record.by_difficulty),
                json.dumps(record.by_category),
                json.dumps(record.metadata),
            ),
        )

        conn.commit()
        conn.close()

    def get_evaluation(self, eval_id: str) -> EvaluationRecord | None:
        """获取单个评估记录"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_record(row)
        return None

    def get_evaluations_by_version(self, version: str) -> list[EvaluationRecord]:
        """获取指定版本的所有评估"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM evaluations WHERE version = ? ORDER BY timestamp DESC", (version,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_record(row) for row in rows]

    def get_evaluations_in_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[EvaluationRecord]:
        """获取时间范围内的评估"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """SELECT * FROM evaluations
               WHERE timestamp BETWEEN ? AND ?
               ORDER BY timestamp ASC""",
            (start_date.isoformat(), end_date.isoformat()),
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_record(row) for row in rows]

    def get_recent_evaluations(self, days: int = 30) -> list[EvaluationRecord]:
        """获取最近 N 天的评估"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.get_evaluations_in_range(start_date, end_date)

    def get_latest_evaluation(self, version: str | None = None) -> EvaluationRecord | None:
        """获取最新的评估记录"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if version:
            cursor.execute(
                "SELECT * FROM evaluations WHERE version = ? ORDER BY timestamp DESC LIMIT 1",
                (version,),
            )
        else:
            cursor.execute("SELECT * FROM evaluations ORDER BY timestamp DESC LIMIT 1")

        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_record(row)
        return None

    def _row_to_record(self, row: tuple) -> EvaluationRecord:
        """将数据库行转换为记录对象"""
        return EvaluationRecord(
            id=row[0],
            version=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            success_rate=row[3],
            total_tasks=row[4],
            avg_latency=row[5],
            pass_at_1=row[6] or 0.0,
            pass_at_3=row[7] or 0.0,
            by_difficulty=json.loads(row[8]) if row[8] else {},
            by_category=json.loads(row[9]) if row[9] else {},
            metadata=json.loads(row[10]) if row[10] else {},
        )


class TrendAnalyzer:
    """
    趋势分析器

    分析评估结果的历史趋势，检测性能回归。
    """

    def __init__(self, store: EvaluationStore):
        self.store = store

    def analyze_trend(
        self, metric: str = "success_rate", days: int = 30, min_data_points: int = 3
    ) -> TrendResult | None:
        """
        分析指标趋势

        Args:
            metric: 要分析的指标 (success_rate, avg_latency, pass_at_1, etc.)
            days: 分析的天数范围
            min_data_points: 最少数据点数量

        Returns:
            TrendResult 或 None（数据不足时）
        """
        records = self.store.get_recent_evaluations(days)

        if len(records) < min_data_points:
            return None

        # 提取指标值
        values = [getattr(r, metric, None) for r in records]
        values = [v for v in values if v is not None]

        if len(values) < min_data_points:
            return None

        # 计算趋势
        current_value = values[-1]
        previous_value = values[0]

        if previous_value == 0:
            change_percent = 100.0 if current_value > 0 else 0.0
        else:
            change_percent = ((current_value - previous_value) / previous_value) * 100

        # 确定方向
        if abs(change_percent) < 5:  # 5% 以内视为稳定
            direction = "stable"
        elif change_percent > 0:
            direction = "up"
        else:
            direction = "down"

        # 计算置信度（基于数据点数量和方差）
        if len(values) >= 5:
            try:
                variance = statistics.variance(values)
                mean = statistics.mean(values)
                cv = (variance**0.5) / mean if mean > 0 else 1
                confidence = max(0.5, 1 - cv)
            except Exception:
                confidence = 0.5
        else:
            confidence = 0.3 + (len(values) * 0.1)

        return TrendResult(
            metric=metric,
            direction=direction,
            change_percent=change_percent,
            current_value=current_value,
            previous_value=previous_value,
            data_points=len(values),
            confidence=min(1.0, confidence),
        )

    def detect_regression(
        self,
        baseline_version: str,
        current_version: str,
        threshold: float = 0.05,
        metrics: list[str] | None = None,
    ) -> list[RegressionResult]:
        """
        检测性能回归

        Args:
            baseline_version: 基线版本
            current_version: 当前版本
            threshold: 回归阈值（默认 5%）
            metrics: 要检查的指标列表

        Returns:
            回归检测结果列表
        """
        if metrics is None:
            metrics = ["success_rate", "avg_latency", "pass_at_1"]

        baseline = self.store.get_latest_evaluation(baseline_version)
        current = self.store.get_latest_evaluation(current_version)

        if not baseline or not current:
            return [
                RegressionResult(
                    has_regression=False,
                    metric="all",
                    baseline_value=0,
                    current_value=0,
                    change_percent=0,
                    threshold=threshold,
                    message="无法获取基线或当前版本的评估数据",
                )
            ]

        results = []
        for metric in metrics:
            baseline_value = getattr(baseline, metric, 0)
            current_value = getattr(current, metric, 0)

            if baseline_value == 0:
                change_percent = 0
            else:
                change_percent = (current_value - baseline_value) / baseline_value

            # 对于延迟，增加是回归；对于其他指标，减少是回归
            if metric == "avg_latency":
                has_regression = change_percent > threshold
            else:
                has_regression = change_percent < -threshold

            message = self._generate_regression_message(
                metric, has_regression, change_percent, threshold
            )

            results.append(
                RegressionResult(
                    has_regression=has_regression,
                    metric=metric,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    change_percent=change_percent * 100,
                    threshold=threshold * 100,
                    message=message,
                )
            )

        return results

    def _generate_regression_message(
        self, metric: str, has_regression: bool, change: float, threshold: float
    ) -> str:
        """生成回归消息"""
        metric_names = {
            "success_rate": "成功率",
            "avg_latency": "平均延迟",
            "pass_at_1": "Pass@1",
            "pass_at_3": "Pass@3",
        }
        name = metric_names.get(metric, metric)

        if has_regression:
            return f"⚠️ {name} 回归: {change * 100:+.1f}% (阈值: {threshold * 100:.1f}%)"
        else:
            return f"✅ {name} 正常: {change * 100:+.1f}%"

    def detect_anomaly(
        self, metric: str = "success_rate", days: int = 30, std_threshold: float = 2.0
    ) -> dict | None:
        """
        检测异常值

        使用标准差方法检测最新值是否为异常。

        Args:
            metric: 要检查的指标
            days: 历史数据天数
            std_threshold: 标准差阈值

        Returns:
            异常检测结果
        """
        records = self.store.get_recent_evaluations(days)

        if len(records) < 5:
            return None

        values = [getattr(r, metric, None) for r in records]
        values = [v for v in values if v is not None]

        if len(values) < 5:
            return None

        mean = statistics.mean(values[:-1])  # 不包括最新值
        std = statistics.stdev(values[:-1])
        latest = values[-1]

        if std == 0:
            z_score = 0
        else:
            z_score = (latest - mean) / std

        is_anomaly = abs(z_score) > std_threshold

        return {
            "metric": metric,
            "is_anomaly": is_anomaly,
            "latest_value": latest,
            "mean": mean,
            "std": std,
            "z_score": z_score,
            "threshold": std_threshold,
            "message": f"{'⚠️ 异常' if is_anomaly else '✅ 正常'}: z-score = {z_score:.2f}",
        }

    def generate_trend_report(self, days: int = 30) -> str:
        """
        生成趋势报告（Markdown 格式）

        Args:
            days: 报告覆盖的天数

        Returns:
            Markdown 格式的报告
        """
        records = self.store.get_recent_evaluations(days)

        report = f"""# 评估趋势报告

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
分析周期: 最近 {days} 天
数据点数: {len(records)}

## 关键指标趋势

"""

        metrics = ["success_rate", "avg_latency", "pass_at_1"]
        metric_names = {
            "success_rate": "成功率",
            "avg_latency": "平均延迟",
            "pass_at_1": "Pass@1",
        }

        for metric in metrics:
            trend = self.analyze_trend(metric, days)
            if trend:
                direction_emoji = {
                    "up": "📈",
                    "down": "📉",
                    "stable": "➡️",
                }
                report += f"""### {metric_names.get(metric, metric)}

- 方向: {direction_emoji.get(trend.direction, "")} {trend.direction}
- 变化: {trend.change_percent:+.1f}%
- 当前值: {trend.current_value:.3f}
- 基线值: {trend.previous_value:.3f}
- 置信度: {trend.confidence:.1%}

"""

        # 异常检测
        report += "## 异常检测\n\n"
        for metric in metrics:
            anomaly = self.detect_anomaly(metric, days)
            if anomaly:
                report += f"- {metric_names.get(metric, metric)}: {anomaly['message']}\n"

        # 按难度分析
        if records:
            latest = records[-1]
            if latest.by_difficulty:
                report += "\n## 按难度分析\n\n"
                report += "| 难度 | 成功率 |\n|------|--------|\n"
                for diff, rate in latest.by_difficulty.items():
                    report += f"| {diff} | {rate:.1%} |\n"

        # 按类别分析
        if records and records[-1].by_category:
            report += "\n## 按类别分析\n\n"
            report += "| 类别 | 成功率 |\n|------|--------|\n"
            for cat, rate in records[-1].by_category.items():
                report += f"| {cat} | {rate:.1%} |\n"

        return report

    def get_performance_summary(self, days: int = 7) -> dict:
        """
        获取性能摘要

        Args:
            days: 摘要覆盖的天数

        Returns:
            性能摘要字典
        """
        records = self.store.get_recent_evaluations(days)

        if not records:
            return {"error": "无评估数据"}

        success_rates = [r.success_rate for r in records]
        latencies = [r.avg_latency for r in records]

        return {
            "period_days": days,
            "total_evaluations": len(records),
            "success_rate": {
                "current": success_rates[-1] if success_rates else 0,
                "avg": statistics.mean(success_rates) if success_rates else 0,
                "min": min(success_rates) if success_rates else 0,
                "max": max(success_rates) if success_rates else 0,
            },
            "latency": {
                "current": latencies[-1] if latencies else 0,
                "avg": statistics.mean(latencies) if latencies else 0,
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
            },
            "trends": {
                "success_rate": self.analyze_trend("success_rate", days),
                "latency": self.analyze_trend("avg_latency", days),
            },
        }


# 便捷函数
def create_evaluation_record(eval_result: dict, version: str = "unknown") -> EvaluationRecord:
    """从评估结果创建记录"""
    return EvaluationRecord(
        id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        version=version,
        timestamp=datetime.now(),
        success_rate=eval_result.get("overall_success_rate", 0),
        total_tasks=eval_result.get("total_tasks", 0),
        avg_latency=eval_result.get("avg_execution_time", 0),
        pass_at_1=eval_result.get("pass_at_1", 0),
        pass_at_3=eval_result.get("pass_at_3", 0),
        by_difficulty=eval_result.get("by_difficulty", {}),
        by_category=eval_result.get("by_category", {}),
        metadata=eval_result.get("metadata", {}),
    )


if __name__ == "__main__":
    # 示例用法
    print("=== 趋势分析器示例 ===\n")

    # 创建存储
    store = EvaluationStore("eval_results/history.db")

    # 创建一些测试数据
    for i in range(5):
        record = EvaluationRecord(
            id=f"test_{i}",
            version="1.0.0",
            timestamp=datetime.now() - timedelta(days=5 - i),
            success_rate=0.8 + i * 0.02,
            total_tasks=100,
            avg_latency=1.5 - i * 0.1,
            pass_at_1=0.75 + i * 0.02,
            by_difficulty={"easy": 0.95, "medium": 0.8, "hard": 0.6},
            by_category={"file": 0.9, "code": 0.85},
        )
        store.save_evaluation(record)

    # 创建分析器
    analyzer = TrendAnalyzer(store)

    # 分析趋势
    trend = analyzer.analyze_trend("success_rate", days=30)
    if trend:
        print(f"成功率趋势: {trend.direction} ({trend.change_percent:+.1f}%)")

    # 检测异常
    anomaly = analyzer.detect_anomaly("success_rate")
    if anomaly:
        print(f"异常检测: {anomaly['message']}")

    # 生成报告
    print("\n" + "=" * 50)
    print(analyzer.generate_trend_report(days=30))
