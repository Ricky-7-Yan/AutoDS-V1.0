document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('analysis-form');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    const taskPlan = document.getElementById('task-plan');
    const executionResults = document.getElementById('execution-results');
    const reportContent = document.getElementById('report-content');
    const visualizationContent = document.getElementById('visualization-content');
    const downloadBtn = document.getElementById('download-report');
    const newQueryBtn = document.getElementById('new-query');
    const tabLinks = document.querySelectorAll('.tab-link');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // 选项卡切换逻辑
    tabLinks.forEach(link => {
        link.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');

            // 更新活动选项卡
            tabLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');

            // 显示对应内容
            tabPanes.forEach(pane => pane.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
        });
    });

    // 表单提交处理
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const query = document.getElementById('query').value.trim();
        if (!query) return;

        // 重置UI
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        error.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('query', query);

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // 显示任务计划
                taskPlan.innerHTML = `
                    <h4>生成的任务计划 (${data.tasks.length}个任务)</h4>
                    <div class="task-list">
                        ${data.tasks.map((task, index) => `
                            <div class="task-item">
                                <h4>任务 ${index + 1}: ${task.task}</h4>
                                <p><strong>工具:</strong> ${task.tool}</p>
                                <p><strong>描述:</strong> ${task.description || '无详细描述'}</p>
                            </div>
                        `).join('')}
                    </div>
                `;

                // 显示执行结果
                executionResults.innerHTML = `
                    <h4>任务执行详情</h4>
                    ${data.results.map(result => `
                        <div class="result-item">
                            <h4>任务 ${result.task_number}: ${result.task_description}</h4>
                            <p><strong>工具:</strong> ${result.tool}</p>
                            <p><strong>结果:</strong> ${result.result}</p>
                        </div>
                    `).join('')}
                `;

                // 显示报告
                reportContent.innerHTML = `
                    <div class="report">
                        ${formatReport(data.report)}
                    </div>
                `;

                // 显示可视化图表
                if (data.chart_urls && data.chart_urls.length > 0) {
                    visualizationContent.innerHTML = `
                        <h4>生成的可视化图表 (${data.chart_urls.length}张)</h4>
                        ${data.chart_urls.map(url => `
                            <div class="chart-container">
                                <img src="${url}" alt="数据可视化图表" class="chart-image">
                            </div>
                        `).join('')}
                    `;
                } else {
                    visualizationContent.innerHTML = '<p>本次分析未生成可视化图表</p>';
                }

                // 设置下载按钮
                if (data.report_path) {
                    const reportName = data.report_path.split('/').pop();
                    downloadBtn.onclick = () => {
                        window.location.href = `/report/${reportName}`;
                    };
                }

                loading.classList.add('hidden');
                results.classList.remove('hidden');

                // 滚动到结果区域
                results.scrollIntoView({ behavior: 'smooth' });
            } else {
                throw new Error(data.error || '未知错误');
            }
        } catch (err) {
            loading.classList.add('hidden');
            error.querySelector('#error-message').textContent = err.message;
            error.classList.remove('hidden');
        }
    });

    // 新查询按钮
    newQueryBtn.addEventListener('click', function() {
        results.classList.add('hidden');
        document.getElementById('query').value = '';
        document.getElementById('query').focus();
    });

    // 格式化报告内容
    function formatReport(report) {
        if (!report) return '<p>无报告内容</p>';

        // 简单的markdown转换
        let formatted = report
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // 粗体
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // 斜体
            .replace(/^# (.*$)/gim, '<h3>$1</h3>') // 一级标题
            .replace(/^## (.*$)/gim, '<h4>$1</h4>') // 二级标题
            .replace(/^### (.*$)/gim, '<h5>$1</h5>') // 三级标题
            .replace(/\n/g, '<br>') // 换行
            .replace(/^- (.*$)/gim, '<li>$1</li>') // 列表项
            .replace(/(<li>.*<\/li>)/g, '<ul>$1</ul>'); // 列表

        return formatted;
    }
});