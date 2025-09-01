import os
import json
import re
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tools import (
    serpapi_search,
    fetch_industry_data,
    create_visualization,
    save_report,
    execute_python_code,
    get_news,
    setup_pinecone,
    store_memory,
    search_memory
)

# 加载环境变量
load_dotenv()


class DataAnalyzerAgent:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_endpoint = os.getenv("DEEPSEEK_API_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")
        self.context = ""
        self.results = []
        self.memory_index = setup_pinecone()
        self.task_counter = 0

    def call_deepseek_api(self, messages, temperature=0.2, max_tokens=2000):
        """调用DeepSeek API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API调用错误: {e}")
            return f"API调用失败: {str(e)}"

    def plan_tasks(self, query: str) -> List[Dict[str, str]]:
        """规划任务步骤"""
        print("规划任务步骤...")

        # 从记忆库中搜索相关经验
        previous_experience = ""
        if self.memory_index:
            memory_results = search_memory(self.memory_index, query, top_k=3)
            if memory_results and 'matches' in memory_results:
                for match in memory_results['matches']:
                    if 'metadata' in match and 'text' in match['metadata']:
                        previous_experience += f"\n先前经验: {match['metadata']['text']}"

        prompt = f"""
        你是一个高级数据分析AI助手。请将以下复杂查询分解为一系列具体的任务步骤。
        考虑之前的经验: {previous_experience}

        查询: {query}

        请以JSON格式返回任务计划，格式如下：
        [
            {{"task": "任务1描述", "tool": "使用的工具", "description": "详细说明"}},
            {{"task": "任务2描述", "tool": "使用的工具", "description": "详细说明"}},
            ...
        ]

        可用的工具包括: 
        - web_search: 用于搜索网络信息
        - fetch_industry_data: 用于获取行业数据
        - get_news: 用于获取最新新闻
        - create_visualization: 用于创建数据可视化
        - execute_python_code: 用于执行数据分析代码
        - save_report: 用于保存最终报告

        请根据查询内容选择合适的工具，并确保任务步骤合理且完整。
        """

        messages = [{"role": "user", "content": prompt}]
        response = self.call_deepseek_api(messages)

        try:
            # 尝试从响应中提取JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                tasks = json.loads(json_match.group())
                return tasks
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")

        # 如果JSON解析失败，返回默认任务
        return [
            {"task": "搜索相关行业信息和发展趋势", "tool": "web_search", "description": "使用网络搜索获取行业最新信息"},
            {"task": "获取行业数据和统计信息", "tool": "fetch_industry_data", "description": "获取结构化行业数据"},
            {"task": "收集相关新闻和市场动态", "tool": "get_news", "description": "获取最新新闻和市场动态"},
            {"task": "分析数据趋势和模式", "tool": "execute_python_code", "description": "使用Python代码进行数据分析"},
            {"task": "创建数据可视化图表", "tool": "create_visualization", "description": "生成可视化图表"},
            {"task": "生成综合分析报告", "tool": "save_report", "description": "整合所有信息生成详细报告"}
        ]

    def execute_task(self, task: Dict[str, str]) -> str:
        """执行单个任务"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"

        task_description = task.get("task", "")
        tool = task.get("tool", "")
        description = task.get("description", "")

        print(f"执行任务 {self.task_counter}: {task_description}")
        print(f"使用工具: {tool}")
        print(f"任务说明: {description}")

        # 根据工具类型执行相应操作
        result = ""

        if tool == "web_search":
            # 提取搜索关键词
            search_query = task_description
            if "搜索" in task_description:
                search_query = task_description.replace("搜索", "").strip()

            search_results = serpapi_search(search_query)
            result = f"搜索完成，找到{len(search_results)}条结果:\n"

            for i, r in enumerate(search_results, 1):
                result += f"{i}. {r['title']}: {r['snippet']}\n"

        elif tool == "fetch_industry_data":
            # 提取行业关键词
            industry = "新能源汽车"  # 默认值
            if "新能源" in task_description:
                industry = "新能源汽车"
            elif "科技" in task_description or "IT" in task_description:
                industry = "科技"
            elif "金融" in task_description:
                industry = "金融"
            elif "医疗" in task_description:
                industry = "医疗"

            data = fetch_industry_data(industry)
            if not data.empty:
                self.context += f"\n获取到的{industry}行业数据:\n{data.head().to_string()}"
                result = f"成功获取{industry}行业数据，共{len(data)}条记录\n数据示例:\n{data.head().to_string()}"
            else:
                result = f"未能获取{industry}行业数据"

        elif tool == "get_news":
            # 提取行业关键词
            industry = "新能源汽车"  # 默认值
            if "新能源" in task_description:
                industry = "新能源汽车"
            elif "科技" in task_description:
                industry = "科技"

            news_results = get_news(industry)
            result = f"获取到{len(news_results)}条相关新闻:\n"

            for i, news in enumerate(news_results, 1):
                result += f"{i}. {news['title']}: {news['description']}\n"

        elif tool == "create_visualization":
            # 尝试从上下文提取数据关键词
            chart_type = "line"
            chart_title = "数据可视化"

            if "趋势" in task_description or "折线" in task_description:
                chart_type = "line"
            elif "柱状" in task_description or "条形" in task_description:
                chart_type = "bar"
            elif "饼" in task_description or "比例" in task_description:
                chart_type = "pie"

            # 这里简化处理，实际应该解析上下文中的数据
            industry = "新能源汽车"
            data = fetch_industry_data(industry)

            chart_path = create_visualization(data, chart_type, chart_title)
            if chart_path:
                self.context += f"\n生成可视化图表: {chart_path}"
                result = f"成功生成{chart_type}类型图表: {chart_path}"
            else:
                result = "生成图表失败"

        elif tool == "execute_python_code":
            # 根据任务描述生成代码
            if "分析" in task_description or "统计" in task_description:
                code = """
# 数据分析示例
import pandas as pd
import numpy as np

# 创建示例数据
dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
sales = np.random.randint(50000, 100000, size=len(dates)) + np.arange(len(dates)) * 2000
growth_rate = np.random.uniform(5, 15, size=len(dates))

data = pd.DataFrame({
    '月份': dates.strftime('%Y-%m'),
    '销量': sales,
    '增长率': growth_rate
})

# 计算统计指标
avg_sales = np.mean(sales)
max_sales = np.max(sales)
min_sales = np.min(sales)
sales_growth = (sales[-1] - sales[0]) / sales[0] * 100

result = f\"平均销量: {avg_sales:.2f}, 最高销量: {max_sales}, 最低销量: {min_sales}\\n总增长率: {sales_growth:.2f}%\"
"""
                execution_result = execute_python_code(code)
                result = f"代码执行结果: {execution_result}"
            else:
                result = "未识别到明确的分析任务"

        elif tool == "save_report":
            # 这里先记录需要保存报告，最后统一处理
            result = "报告将在此任务完成后保存"

        else:
            # 如果没有匹配的工具或任务，使用API处理
            prompt = f"""
            根据当前上下文和任务描述，执行以下任务：
            任务: {task_description}
            任务说明: {description}

            上下文信息: {self.context}

            请根据任务要求执行相应的操作，并返回任务执行结果。
            """

            messages = [{"role": "user", "content": prompt}]
            result = self.call_deepseek_api(messages)

        # 存储任务结果到记忆库
        if self.memory_index:
            store_memory(
                self.memory_index,
                f"任务: {task_description}, 结果: {result}",
                task_id,
                {"type": "task_execution", "tool": tool}
            )

        return result

    def generate_report(self, query: str) -> str:
        """生成最终报告"""
        print("\n生成最终报告...")

        # 从记忆库中获取相关任务结果
        task_context = ""
        if self.memory_index:
            memory_results = search_memory(self.memory_index, query, top_k=10)
            if memory_results and 'matches' in memory_results:
                for match in memory_results['matches']:
                    if 'metadata' in match and 'text' in match['metadata']:
                        task_context += f"\n{match['metadata']['text']}"

        prompt = f"""
        根据以下查询和分析结果，生成一份专业的数据分析报告：

        查询: {query}

        分析结果和上下文: {self.context}

        任务执行记录: {task_context}

        请生成一份结构完整、内容详实的报告，包括：
        1. 执行摘要和主要发现
        2. 数据来源和方法论
        3. 详细数据分析和趋势
        4. 可视化解读（如果有）
        5. 结论和建议
        6. 局限性和未来研究方向

        报告应该专业、客观，并基于提供的数据和分析结果。使用markdown格式组织内容。
        """

        messages = [{"role": "user", "content": prompt}]
        report = self.call_deepseek_api(messages, temperature=0.7, max_tokens=3000)
        return report

    def run(self, query: str) -> Dict[str, Any]:
        """运行自主Agent处理查询"""
        print(f"开始处理查询: {query}")

        # 重置上下文和结果
        self.context = f"查询: {query}\n"
        self.results = []
        self.task_counter = 0

        # 规划任务
        tasks = self.plan_tasks(query)
        print(f"生成的任务计划: {tasks}")

        # 执行任务
        for i, task in enumerate(tasks, 1):
            print(f"\n执行任务 {i}/{len(tasks)}")
            result = self.execute_task(task)
            print(f"任务结果: {result}")

            # 更新上下文和结果
            self.context += f"\n任务 {i}: {task['task']}\n结果: {result}"
            self.results.append({
                "task_number": i,
                "task_description": task['task'],
                "tool": task.get('tool', ''),
                "description": task.get('description', ''),
                "result": result
            })

        # 生成最终报告
        report = self.generate_report(query)

        # 保存报告
        report_path = save_report(report)

        # 查找可视化图表
        chart_paths = []
        if os.path.exists('static/charts'):
            for file in os.listdir('static/charts'):
                if file.startswith('chart_') and file.endswith('.png'):
                    chart_paths.append(f"static/charts/{file}")

        # 存储最终报告到记忆库
        if self.memory_index:
            store_memory(
                self.memory_index,
                f"完整分析报告: {report[:500]}...",
                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                {"type": "final_report", "query": query}
            )

        return {
            "query": query,
            "tasks": tasks,
            "results": self.results,
            "report": report,
            "report_path": report_path,
            "chart_paths": chart_paths
        }


# 单例Agent实例
agent_instance = DataAnalyzerAgent()