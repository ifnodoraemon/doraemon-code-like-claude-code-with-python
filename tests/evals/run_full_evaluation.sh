#!/bin/bash
# 完整评估运行脚本 - 使用新的完整评估系统

set -e  # 遇到错误立即退出

echo "================================================================================"
echo "Doraemon Code Agent 完整评估系统 v2.0"
echo "================================================================================"

# 解析命令行参数
PARALLEL=true
MAX_WORKERS=4
N_TRIALS=1
USE_LLM_JUDGE=false
MODE="full"  # full, baseline, comparison

while [[ $# -gt 0 ]]; do
    case $1 in
        --serial)
            PARALLEL=false
            shift
            ;;
        --workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --llm-judge)
            USE_LLM_JUDGE=true
            shift
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --serial          使用串行评估 (默认: 并行)"
            echo "  --workers N       并行工作数 (默认: 4)"
            echo "  --trials N        每个任务运行次数 (默认: 1)"
            echo "  --llm-judge       启用 LLM 评判 (默认: 关闭)"
            echo "  --mode MODE       评估模式: full, baseline, comparison (默认: full)"
            echo "  --help            显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                    # 默认配置"
            echo "  $0 --workers 8 --trials 3             # 8 并行, 3 次试验"
            echo "  $0 --llm-judge --mode baseline        # 基线评估 + LLM 评判"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p eval_results/{comprehensive,baseline,parallel,metrics,model_comparison}

echo ""
echo "评估配置:"
echo "  模式: $MODE"
echo "  并行: $PARALLEL"
echo "  工作数: $MAX_WORKERS"
echo "  试验次数: $N_TRIALS"
echo "  LLM 评判: $USE_LLM_JUDGE"
echo ""

# 运行评估
case $MODE in
    full)
        echo "运行完整评估..."
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from tests.evals.comprehensive_evaluator import ComprehensiveEvaluator

# 创建评估器
evaluator = ComprehensiveEvaluator(
    output_dir='eval_results/comprehensive',
    parallel=$PARALLEL,
    max_workers=$MAX_WORKERS,
    n_trials=$N_TRIALS,
    use_llm_judge=$USE_LLM_JUDGE
)

# 注意: 需要实现 agent_factory
# 这里提供一个示例框架
def create_agent():
    # TODO: 实现 Agent 创建逻辑
    # from src.host.cli.main import DoraemonAgent
    # return DoraemonAgent()
    print('警告: Agent 工厂未实现，使用模拟 Agent')
    from tests.evals.test_evaluation_system import MockAgent
    return MockAgent()

# 运行评估
try:
    report = evaluator.run_full_evaluation(create_agent)
    print('\n✅ 评估完成！')
    print(f'成功率: {report[\"metrics\"][\"core_metrics\"][\"success_rate\"]*100:.1f}%')
except Exception as e:
    print(f'\n❌ 评估失败: {e}')
    sys.exit(1)
"
        ;;

    baseline)
        echo "运行基线评估..."
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from tests.evals.comprehensive_evaluator import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(
    output_dir='eval_results/baseline',
    parallel=$PARALLEL,
    max_workers=$MAX_WORKERS,
    n_trials=$N_TRIALS,
    use_llm_judge=$USE_LLM_JUDGE
)

def create_agent():
    from tests.evals.test_evaluation_system import MockAgent
    return MockAgent()

try:
    baseline = evaluator.run_baseline_evaluation(create_agent)
    print('\n✅ 基线评估完成！')
except Exception as e:
    print(f'\n❌ 基线评估失败: {e}')
    sys.exit(1)
"
        ;;

    comparison)
        echo "运行多模型对比..."
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from tests.evals.comprehensive_evaluator import ComprehensiveEvaluator
from tests.evals.test_evaluation_system import MockAgent

evaluator = ComprehensiveEvaluator(
    output_dir='eval_results/model_comparison',
    parallel=$PARALLEL,
    max_workers=$MAX_WORKERS,
    n_trials=$N_TRIALS,
    use_llm_judge=$USE_LLM_JUDGE
)

# 创建不同模型的 Agent 工厂
def create_model_a():
    return MockAgent(success_rate=0.9)

def create_model_b():
    return MockAgent(success_rate=0.8)

def create_model_c():
    return MockAgent(success_rate=0.85)

try:
    comparison = evaluator.run_model_comparison({
        'Model A (90%)': create_model_a,
        'Model B (80%)': create_model_b,
        'Model C (85%)': create_model_c
    })
    print('\n✅ 模型对比完成！')
except Exception as e:
    print(f'\n❌ 模型对比失败: {e}')
    sys.exit(1)
"
        ;;

    *)
        echo "错误: 未知模式 '$MODE'"
        echo "支持的模式: full, baseline, comparison"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "评估完成！"
echo "================================================================================"
echo ""
echo "结果保存在:"
echo "  - eval_results/comprehensive/  (完整评估)"
echo "  - eval_results/baseline/       (基线评估)"
echo "  - eval_results/parallel/       (并行评估)"
echo "  - eval_results/metrics/        (详细指标)"
echo "  - eval_results/model_comparison/ (模型对比)"
echo ""
echo "查看报告:"
echo "  - JSON: eval_results/comprehensive/report_*.json"
echo "  - Markdown: eval_results/comprehensive/report_*.md"
echo ""
