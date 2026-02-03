#!/bin/bash
# 完整评估运行脚本

echo "=========================================="
echo "Doraemon Code Agent 完整评估"
echo "=========================================="

# 创建输出目录
mkdir -p eval_results/{baseline,model_comparison,reports}

# 1. 运行基线评估
echo -e "\n[1/4] 运行基线评估..."
python tests/evals/baseline_runner.py

# 2. 运行 LLM-as-Judge 评估
echo -e "\n[2/4] 运行 LLM-as-Judge 评估..."
python tests/evals/llm_judge_evaluator.py

# 3. 运行多模型对比
echo -e "\n[3/4] 运行多模型对比..."
python tests/evals/multi_model_comparator.py

# 4. 生成综合报告
echo -e "\n[4/4] 生成综合报告..."
python -c "
from tests.evals.baseline_runner import BaselineRunner
from tests.evals.multi_model_comparator import MultiModelComparator

# 加载结果
runner = BaselineRunner()
baseline = runner.load_baseline()

if baseline:
    runner.print_baseline_report(baseline['summary'])
else:
    print('未找到基线数据')
"

echo -e "\n=========================================="
echo "评估完成！"
echo "结果保存在 eval_results/ 目录"
echo "=========================================="
