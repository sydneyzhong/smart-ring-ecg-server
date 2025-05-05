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

@app.route('/analyze_and_show')
def analyze_and_show():
    try:
        processor = ECGProcessor()
        filepath = '/tmp/uploads/test.dat'  # 使用容器内的绝对路径
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Test file not found at {filepath}")
        
        success, analysis_results, report = processor.analyze_ecg_file(filepath)
        if not success:
            raise Exception("ECG analysis failed")

        # 生成图片
        plt.figure(figsize=(12,4))
        ecg_signal = analysis_results['basic_info']['ecg_signal'][:1000]
        plt.plot(ecg_signal)
        plt.title('ECG Signal')
        
        web_img_path = 'static/ecg_web.png'
        plt.savefig(web_img_path)
        plt.close()

        return render_template('analysis_report.html',
                             basic_info={
                                 'filename': analysis_results['basic_info']['filename'],
                                 'duration': analysis_results['basic_info']['duration'],
                                 'fs': analysis_results['basic_info']['fs']
                             },
                             wave_features=analysis_results['wave_features'],
                             hrv_analysis=analysis_results['hrv_analysis'],
                             plot_filename='ecg_web.png')
    except Exception as e:
        app.logger.error(f"Error in analyze_and_show: {str(e)}")
        return f"Internal Server Error: {str(e)}", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# 其他路由保持不变...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)