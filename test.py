# from ecg_processor import ECGProcessor
# processor = ECGProcessor()
# success, results, report = processor.analyze_ecg_file("test.dat")
# print(f"Success: {success}")
# print(f"Health Index: {results.get('health_index')}")
from ecg_processor import ECGProcessor
from flask import Flask, send_file
import os

app = Flask(__name__)

# 初始化处理器
processor = ECGProcessor()

@app.route('/report')
def show_report():
    # 分析数据文件
    success, results, report = processor.analyze_ecg_file("/tmp/uploads/test.dat")  # 注意容器内路径
    
    if not success:
        return f"Analysis failed: {results['error']}", 500
    
    # 返回生成的HTML报告
    return send_file(report['html_report'])

if __name__ == '__main__':
    # 启动临时Web服务（端口5001避免冲突）
    app.run(host='0.0.0.0', port=5001)