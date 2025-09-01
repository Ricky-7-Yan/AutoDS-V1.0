import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# 加载环境变量
load_dotenv()

# 设置可视化中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def serpapi_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    使用SERPAPI进行真实网络搜索
    返回搜索结果的列表
    """
    print(f"执行真实搜索: {query}")

    try:
        # SERPAPI调用
        params = {
            'q': query,
            'api_key': os.getenv('SERPAPI_API_KEY'),
            'engine': 'google',
            'num': num_results,
            'hl': 'zh-cn',
            'gl': 'cn'
        }

        response = requests.get('https://serpapi.com/search', params=params)
        response.raise_for_status()
        results = response.json()

        # 解析搜索结果
        search_results = []
        if 'organic_results' in results:
            for item in results['organic_results']:
                search_results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })

        return search_results[:num_results]

    except Exception as e:
        print(f"SERPAPI搜索错误: {e}")
        # 备用方案：模拟搜索
        return web_search(query, num_results)


def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    模拟Web搜索（备用方案）
    """
    print(f"执行模拟搜索: {query}")
    mock_results = [
        {"title": "2023-2024年新能源汽车行业分析报告", "url": "https://example.com/report1",
         "snippet": "2023年新能源汽车销量同比增长35%，市场渗透率达到28%。2024年预计增长25%。"},
        {"title": "新能源汽车电池技术发展趋势白皮书", "url": "https://example.com/report2",
         "snippet": "固态电池技术预计将在2025年实现商业化，能量密度提升40%以上。"},
        {"title": "全球新能源汽车市场格局分析", "url": "https://example.com/report3",
         "snippet": "中国占据全球新能源汽车市场份额的45%，欧洲占30%，北美占15%。"},
        {"title": "新能源汽车政策支持与补贴指南", "url": "https://example.com/report4",
         "snippet": "各国政府继续推出新能源汽车补贴政策，中国延长购置税减免至2025年。"},
        {"title": "电动汽车充电基础设施发展报告", "url": "https://example.com/report5",
         "snippet": "全球充电桩数量同比增长50%，快充技术取得重大突破。"},
    ]
    return mock_results[:num_results]


def fetch_industry_data(industry: str, period: str = "1y") -> pd.DataFrame:
    """
    获取行业数据（使用AlphaVantage API）
    """
    print(f"获取行业数据: {industry}")

    try:
        # 这里使用模拟数据，实际应使用AlphaVantage API
        # 模拟数据生成
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')

        if "新能源" in industry or "汽车" in industry:
            sales = np.random.randint(50000, 100000, size=len(dates)) + np.arange(len(dates)) * 2000
            market_share = np.linspace(20, 35, len(dates))
            growth_rate = np.random.uniform(5, 15, size=len(dates))

            data = pd.DataFrame({
                '日期': dates,
                '销量': sales,
                '市场份额': market_share,
                '增长率': growth_rate
            })
        elif "科技" in industry or "IT" in industry:
            revenue = np.random.randint(100000, 500000, size=len(dates)) + np.arange(len(dates)) * 5000
            market_share = np.linspace(15, 25, len(dates))
            growth_rate = np.random.uniform(8, 20, size=len(dates))

            data = pd.DataFrame({
                '日期': dates,
                '收入': revenue,
                '市场份额': market_share,
                '增长率': growth_rate
            })
        else:
            # 默认行业数据
            values = np.random.randint(50000, 150000, size=len(dates)) + np.arange(len(dates)) * 3000
            market_share = np.linspace(10, 20, len(dates))
            growth_rate = np.random.uniform(3, 12, size=len(dates))

            data = pd.DataFrame({
                '日期': dates,
                '数值': values,
                '市场份额': market_share,
                '增长率': growth_rate
            })

        return data

    except Exception as e:
        print(f"获取行业数据错误: {e}")
        return pd.DataFrame()


def get_news(industry: str, num_articles: int = 5) -> List[Dict[str, str]]:
    """
    使用NewsAPI获取行业新闻
    """
    print(f"获取新闻: {industry}")

    try:
        # 模拟新闻数据
        mock_news = [
            {
                'title': f'{industry}行业迎来政策利好，多家企业受益',
                'url': 'https://example.com/news1',
                'description': f'近日，国家出台多项政策支持{industry}行业发展，预计将带动相关企业业绩增长。',
                'publishedAt': '2024-01-15T10:00:00Z'
            },
            {
                'title': f'{industry}技术创新取得突破，市场前景广阔',
                'url': 'https://example.com/news2',
                'description': f'{industry}领域最新技术突破将改变行业格局，多家机构看好未来发展。',
                'publishedAt': '2024-01-10T14:30:00Z'
            },
            {
                'title': f'专家解读{industry}行业发展趋势',
                'url': 'https://example.com/news3',
                'description': f'行业专家分析认为，{industry}行业将在未来三年保持高速增长态势。',
                'publishedAt': '2024-01-05T09:15:00Z'
            }
        ]

        return mock_news[:num_articles]

    except Exception as e:
        print(f"获取新闻错误: {e}")
        return []


def create_visualization(data: pd.DataFrame, chart_type: str = "line", title: str = "数据可视化") -> str:
    """
    创建数据可视化图表并保存为图片
    返回图片文件路径
    """
    if data.empty:
        return ""

    # 确保输出目录存在
    os.makedirs('static/charts', exist_ok=True)

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"static/charts/chart_{timestamp}.png"

    # 创建图表
    plt.figure(figsize=(12, 7))

    try:
        if chart_type == "line" and '日期' in data.columns:
            for column in data.columns:
                if column != '日期' and data[column].dtype in [np.int64, np.float64]:
                    plt.plot(data['日期'], data[column], label=column, marker='o')

            plt.title(f'{title} - 趋势图')
            plt.xlabel('日期')
            plt.ylabel('数值')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)

        elif chart_type == "bar" and '日期' in data.columns:
            # 选择数值型列
            numeric_columns = [col for col in data.columns if
                               col != '日期' and data[col].dtype in [np.int64, np.float64]]

            if numeric_columns:
                # 使用最后一个数值列
                column = numeric_columns[-1]
                plt.bar(data['日期'].astype(str), data[column])
                plt.title(f'{title} - 柱状图')
                plt.xlabel('日期')
                plt.ylabel(column)
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)

        elif chart_type == "pie":
            # 选择最后一个数值列
            numeric_columns = [col for col in data.columns if data[col].dtype in [np.int64, np.float64]]
            if numeric_columns:
                column = numeric_columns[-1]
                # 使用最后一条数据
                last_values = data.iloc[-1][numeric_columns]
                plt.pie(last_values, labels=numeric_columns, autopct='%1.1f%%')
                plt.title(f'{title} - 占比图')

        else:
            # 默认图表：选择所有数值列绘制趋势图
            numeric_columns = [col for col in data.columns if data[col].dtype in [np.int64, np.float64]]
            if '日期' in data.columns and numeric_columns:
                for column in numeric_columns:
                    plt.plot(data['日期'], data[column], label=column, marker='o')

                plt.title(f'{title} - 多指标趋势图')
                plt.xlabel('日期')
                plt.ylabel('数值')
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)
            elif numeric_columns:
                # 如果没有日期列，绘制柱状图
                plt.bar(range(len(data)), data[numeric_columns[0]])
                plt.title(f'{title} - 柱状图')
                plt.xlabel('样本')
                plt.ylabel(numeric_columns[0])
                plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    except Exception as e:
        print(f"创建可视化错误: {e}")
        return ""


def save_report(report_content: str, filename: Optional[str] = None) -> str:
    """
    保存报告内容到文件
    返回文件路径
    """
    os.makedirs('outputs', exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/report_{timestamp}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return filename


def execute_python_code(code: str) -> str:
    """
    执行Python代码并返回结果
    注意：在实际应用中需要谨慎使用，可能存在安全风险
    """
    try:
        # 创建一个局部命名空间来执行代码
        local_vars = {}

        # 添加安全模块到全局变量
        safe_globals = {
            '__builtins__': {
                'print': print,
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'set': set,
                'round': round,
                'sum': sum,
                'min': min,
                'max': max,
            },
            'pd': pd,
            'np': np,
            'plt': plt
        }

        # 执行代码
        exec(code, safe_globals, local_vars)

        # 尝试获取有意义的输出
        output = local_vars.get('result', '代码执行成功，但没有返回结果')

        # 如果有图表被创建，保存它们
        if 'plt' in safe_globals and safe_globals['plt'].get_fignums():
            os.makedirs('static/charts', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = f"static/charts/chart_{timestamp}.png"
            safe_globals['plt'].savefig(chart_path)
            safe_globals['plt'].close('all')
            return f"代码执行成功，生成图表: {chart_path}\n输出: {output}"

        return f"代码执行成功:\n{output}"

    except Exception as e:
        return f"代码执行错误: {str(e)}"


def setup_pinecone():
    """
    初始化Pinecone向量数据库
    """
    try:
        import pinecone

        # 初始化Pinecone
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
        )

        # 创建或获取索引
        index_name = "autogpt-memory"

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI嵌入维度
                metric="cosine"
            )

        return pinecone.Index(index_name)

    except Exception as e:
        print(f"Pinecone初始化错误: {e}")
        return None


def store_memory(index, text: str, task_id: str, metadata: Optional[Dict[str, Any]] = None):
    """
    存储记忆到Pinecone
    """
    try:
        # 生成嵌入向量 (这里简化处理，实际应使用OpenAI API)
        # 实际使用时应该调用OpenAI的嵌入API
        embedding = np.random.rand(1536).tolist()

        # 准备元数据
        if metadata is None:
            metadata = {}

        metadata.update({
            'task_id': task_id,
            'text': text,
            'timestamp': datetime.now().isoformat()
        })

        # 存储到Pinecone
        index.upsert([(task_id, embedding, metadata)])

        return True

    except Exception as e:
        print(f"存储记忆错误: {e}")
        return False


def search_memory(index, query: str, top_k: int = 5):
    """
    从Pinecone搜索相关记忆
    """
    try:
        # 生成查询嵌入向量 (这里简化处理，实际应使用OpenAI API)
        query_embedding = np.random.rand(1536).tolist()

        # 查询Pinecone
        results = index.query(
            queries=[query_embedding],
            top_k=top_k,
            include_metadata=True
        )

        return results

    except Exception as e:
        print(f"搜索记忆错误: {e}")
        return None