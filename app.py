
import numpy as np
from datetime import datetime
import os
import time
import hashlib
from flask import Flask, render_template, send_from_directory, request, jsonify
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ecg_processor import ECGProcessor

app = Flask(__name__,
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

# 配置常量
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'dat', 'csv'}
APP_CONFIG = {'app1': {'secret': 'ECG_Service_Secret_2025!'}}

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('reports', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_signature(params):
    try:
        app_id = params['appId']
        timestamp = params['time']
        timestamp_seconds = float(timestamp) / 1000 if len(timestamp) > 10 else float(timestamp)
        if abs(time.time() - timestamp_seconds) > 300:
            return False

        device_id = params['id']
        sign = params['sign']
        app_secret = APP_CONFIG.get(app_id, {}).get('secret', '')
        raw_str = f"{app_id}|{timestamp}|{device_id}|{app_secret}"
        return sign == hashlib.md5(raw_str.encode()).hexdigest()
    except Exception:
        return False
    

def generate_mobile_report_data(results):
    """生成移动端专用报告数据"""
    # 1. 基础评分计算
    #score = calculate_health_score(results)
    score = int(round(calculate_health_score(results)))
    
    # 2. 核心指标
    core_metrics = {
        '平均心率': f"{results.get('heart_rate', 73)} 次/分钟",
        '心率变异性': f"{round(results['hrv_analysis']['rmssd'], 1)} 毫秒",  # 保留1位小数
        '心律': results['arrhythmia']['conclusion'],
        'QT间期': f"{results['wave_features']['qtc']} 毫秒"
    }    

    # 3. 疾病风险表
    risk_table = [
        ["急性心肌梗死", "极高", "10.0/10", "冠状动脉急性闭塞导致心肌坏死"],
        ["心房颤动", "低", "2.3/10", "P波异常提示潜在风险"],
        ["室性心动过速", "中", "5.7/10", "QRS波宽度正常但需关注"]
    ]
    
    # 4. 专业解读点
    interpretations = [
        f"自主神经系统: {results['hrv_analysis']['assessment']}",
        f"P波状态: {results['wave_features']['p_waves']['assessment']}",
        f"T波状态: {results['wave_features']['t_waves']['assessment']}",
        "ST段状态: 正常",
        f"节律异常: {results['arrhythmia']['conclusion']}"
    ]
    
    # 5. 健康建议
    recommendations = [
        "保持规律作息，每天7-8小时睡眠",
        "每周进行3-5次中等强度有氧运动",
        "减少咖啡因和酒精摄入",
        "建议3个月后复查"
    ]
    
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'score': score,
        'evaluation': "良好" if score >= 70 else "需关注",  # 添加评价字段
        'core_metrics': core_metrics,
        # ... 其他数据保持不变 ...
        'risk_table': risk_table,
        'interpretations': interpretations,
        'recommendations': recommendations
    }

def calculate_health_score(results):
    """计算健康评分(示例逻辑)"""
    base = 80
    # 根据HRV调整
    base -= max(0, 25 - results['hrv_analysis']['rmssd']) 
    # 根据异常情况调整
    if "异常" in results['wave_features']['p_waves']['assessment']:
        base -= 5
    if "异常" in results['wave_features']['t_waves']['assessment']:
        base -= 5
    return max(50, min(100, base))

@app.route('/analyze_and_show')
def analyze_and_show():
    # 新增容器内多路径检查
    possible_paths = [
        '/app/data/test.dat',          # Docker映射路径
        '/tmp/uploads/test.dat',       # 临时上传路径
        '/home/ubuntu/smart-ring-ecg-server/data/test.dat'  # 宿主机路径
    ]
    
    valid_file = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.getsize(path) > 1024:  # 文件需大于1KB
            valid_file = path
            break
            
    if not valid_file:
        return "未找到有效数据文件", 404
        
    try:
        processor = ECGProcessor()
        success, results, _ = processor.analyze_ecg_file(valid_file)
        
        # 添加心率验证
        heart_rate = results.get('heart_rate')
        if not heart_rate or heart_rate < 30 or heart_rate > 200:
            return "心率计算异常，请检查输入数据", 500
            
        # 生成简化版报告
        return f"""
        <h2>ECG分析结果</h2>
        <p>文件: {os.path.basename(test_file)}</p>
        #<p>心率: {results.get('heart_rate', 'N/A')} BPM</p>
        <p>心率: {results.get('heart_rate', results.get('wave_features', {}).get('qrs_complex', {}).get('count', 'N/A'))} BPM</p>
        <p>HRV-RMSSD: {results.get('hrv_analysis', {}).get('rmssd', 'N/A')} ms</p>
        <p>分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
    except Exception as e:
        return f"服务器错误: {str(e)}", 500

def generate_ecg_plot(signal):
    """生成移动端优化的ECG图"""
    plt.figure(figsize=(10, 3), dpi=80)  # 更适合手机的尺寸
    plt.plot(signal, linewidth=1)
    plt.axis('off')  # 移除坐标轴
    plt.savefig('static/ecg_mobile.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_radar_chart(results):
    """生成移动端雷达图"""
    labels = ['心率', 'P波', 'QRS波', 'T波', 'HRV']
    values = [
        min(1, results.get('heart_rate', 70)/100),
        1 - abs(results['wave_features']['p_waves']['average_pr_interval']/200),
        results['wave_features']['qrs_complex']['count']/150,
        1 - abs(results['wave_features']['t_waves']['average_qt_interval']-300)/100,
        results['hrv_analysis']['rmssd']/30
    ]
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=1)
    ax.fill(angles, values, alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    plt.savefig('static/radar_mobile.png', bbox_inches='tight')
    plt.close()


@app.route('/')
def index():
    return """
    <h1>ECG Analysis Service</h1>
    <p>Available endpoints:</p>
    <ul>
        <li>POST /api/analyze - ECG analysis endpoint</li>
        <li>GET /api/health - Health check</li>
    </ul>
    """
# 确保静态路由正确定义
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        filename,
        mimetype='image/png' if filename.endswith('.png') else None)

# app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('/app/static', filename)

@app.route('/upload', methods=['GET'])
def upload_form():
    return '''
    <h1>ECG文件上传</h1>
    <form action="/api/analyze" method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="hidden" name="appId" value="app1">
      <input type="hidden" name="id" value="device123">
      <input type="hidden" name="servertype" value="ECG">
      <input type="hidden" name="time" id="timestamp">
      <input type="hidden" name="sign" id="signature">
      <button type="submit">提交分析</button>
    </form>
    <script>
      document.querySelector('form').addEventListener('submit', function(e) {
        const timestamp = Math.floor(Date.now()/1000);
        document.getElementById('timestamp').value = timestamp;
        document.getElementById('signature').value = md5(`app1|${timestamp}|device123|ECG_Service_Secret_2025!`);
      });
    </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/blueimp-md5/2.19.0/js/md5.min.js"></script>
    '''

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "version": "1.0"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
