from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from datetime import datetime
from agent import agent_instance
from tools import save_report

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        query = request.form.get('query', '').strip()

        if not query:
            return jsonify({"error": "请输入查询内容"}), 400

        # 运行Agent处理查询
        result = agent_instance.run(query)

        # 确保输出目录存在
        os.makedirs('outputs', exist_ok=True)

        # 保存JSON格式的完整结果
        result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"outputs/{result_filename}", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 准备图表URL
        chart_urls = []
        for chart_path in result.get('chart_paths', []):
            chart_name = os.path.basename(chart_path)
            chart_urls.append(f"/static/charts/{chart_name}")

        response_data = {
            "query": result['query'],
            "tasks": result['tasks'],
            "results": result['results'],
            "report": result['report'],
            "report_path": result['report_path'],
            "chart_urls": chart_urls,
            "success": True
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"处理请求时发生错误: {str(e)}", "success": False}), 500


@app.route('/report/<filename>')
def get_report(filename):
    try:
        return send_file(f"outputs/{filename}", as_attachment=True)
    except FileNotFoundError:
        return "文件未找到", 404


@app.route('/static/charts/<filename>')
def get_chart(filename):
    try:
        return send_file(f"static/charts/{filename}", mimetype='image/png')
    except FileNotFoundError:
        return "图片未找到", 404


@app.route('/api/tasks', methods=['GET'])
def get_recent_tasks():
    """获取最近的任务记录"""
    try:
        tasks = []
        if os.path.exists('outputs'):
            json_files = [f for f in os.listdir('outputs') if f.endswith('.json')]
            json_files.sort(reverse=True)

            for file in json_files[:5]:  # 最近5个文件
                with open(f"outputs/{file}", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    tasks.append({
                        'query': data.get('query', ''),
                        'timestamp': file.replace('result_', '').replace('.json', ''),
                        'report_path': data.get('report_path', '')
                    })

        return jsonify({"tasks": tasks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 确保静态文件目录存在
    os.makedirs('static/charts', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    app.run(debug=True, port=5000, host='0.0.0.0')