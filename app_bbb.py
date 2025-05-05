from flask import send_file
from flask import Flask, request, jsonify
from flask import Flask, send_from_directory
from flask import Flask, render_template
from flask import Flask, render_template, url_for

import hashlib
import time
import os
from werkzeug.utils import secure_filename
from ecg_processor import ECGProcessor  # 您的ECG处理类

import matplotlib
matplotlib.use('Agg')  # 确保在无GUI环境下可用
import matplotlib.pyplot as plt

app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')


# 确保目录存在
os.makedirs('/app/data/ecg_uploads', exist_ok=True)
os.makedirs('/app/static', exist_ok=True)
os.makedirs('/app/reports', exist_ok=True)

# 认证配置（实际应从环境变量读取）
APP_CONFIG = {
    'app1': {'secret': 'ECG_Service_Secret_2025!'}
}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
def verify_signature(params):
    try:
        print(f"\n[DEBUG] Received params: {params}")
        app_id = params['appId']
        timestamp = params['time']  # 明确定义timestamp
        
        # 统一时间戳处理（兼容秒级和毫秒级）
        timestamp_seconds = float(timestamp) / 1000 if len(timestamp) > 10 else float(timestamp)
        if abs(time.time() - timestamp_seconds) > 300:
            print(f"[ERROR] Timestamp expired: {timestamp}")
            return False

        device_id = params['id']
        sign = params['sign']
        app_secret = APP_CONFIG.get(app_id, {}).get('secret', '')
        
        raw_str = f"{app_id}|{timestamp}|{device_id}|{app_secret}"
        server_sign = hashlib.md5(raw_str.encode()).hexdigest()
        
        print(f"[DEBUG] Client sign: {sign}")
        print(f"[DEBUG] Server sign: {server_sign}")
        
        return sign == server_sign
    except Exception as e:
        print(f"[ERROR] Auth failed: {str(e)}")
        return False


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('/app/static', filename)

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

# 修改analyze_and_show路由
@app.route('/analyze_and_show_before')
def analyze_and_show_before():
    try:
        processor = ECGProcessor()
        filepath = "/tmp/uploads/test.dat"
        
        # 验证文件存在
        if not os.path.exists(filepath):
            return f"数据文件不存在: {filepath}", 404
            
        # 处理ECG数据
        success, results, report = processor.analyze_ecg_file(filepath)
        if not success:
            return f"分析失败: {results}", 500
            
        # 验证报告生成
        if not report or not os.path.exists(report.get('html_report', '')):
            return "报告生成失败", 500
            
        return send_file(report['html_report'])
        
    except Exception as e:
        return f"服务器错误: {str(e)}", 500

@app.route('/analyze_and_show')
def analyze_and_show():
    try:
        # 1. 您的分析逻辑
        analysis_results = generate_ecg_analysis()
        
        # 2. 生成图片
        plt.figure(figsize=(12,4))
        plt.plot(analysis_results['basic_info']['ecg_signal'][:1000])  # 示例数据
        plt.title('ECG Signal')
        
        # 确保目录存在
        os.makedirs('static', exist_ok=True)
        web_img_path = 'static/ecg_web.png'
        plt.savefig(web_img_path)
        plt.close()
        
        # 3. 准备模板数据
        template_data = {
            'basic_info': {
                'filename': analysis_results['basic_info']['filename'],
                'duration': analysis_results['basic_info']['duration'],
                'fs': analysis_results['basic_info']['fs']
            },
            'wave_features': analysis_results['wave_features'],
            'hrv_analysis': analysis_results['hrv_analysis'],
            'plot_filename': 'ecg_web.png'
        }
        
        return render_template('analysis_report.html', **template_data)
        
    except Exception as e:
        app.logger.error(f"Error in analyze_and_show: {str(e)}")
        return "Internal Server Error", 500
       
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

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "version": "1.0"})

@app.route('/api/analyze', methods=['POST'])  # 取消注释并修复此路由
def analyze():
    # 验证参数
    required_fields = ['appId', 'time', 'id', 'sign', 'servertype']
    if not all(field in request.form for field in required_fields):
        return jsonify({'code': 400, 'message': 'Missing parameters'})
    
    if not verify_signature(request.form):
        return jsonify({'code': 403, 'message': 'Authentication failed'})
    
    # 检查文件
    if 'file' not in request.files:
        return jsonify({'code': 400, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'code': 400, 'message': 'Invalid file'})
    
    # 处理请求
    server_type = request.form['servertype']
    if server_type == 'ECG':
        try:
            # 保存文件
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # 调用ECG处理器
            processor = ECGProcessor()
            success, results, report = processor.analyze_ecg_file(filepath)
            
            if success:
                return jsonify({
                    'code': 200,
                    'data': {
                        'report': results,
                        'html_path': report['html_report'] if report else None
                    }
                })
            else:
                return jsonify({'code': 500, 'message': results.get('error', 'Analysis failed')})
        except Exception as e:
            return jsonify({'code': 500, 'message': str(e)})
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({'code': 400, 'message': f'Unsupported server type: {server_type}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # 添加debug模式