import numpy as np
from ecg_processor import ECGProcessor

def debug_analysis(filepath):
    print("\n===== 开始调试分析 =====")
    
    # 加载原始数据
    raw_signal = np.fromfile(filepath, dtype='>i2')
    print(f"原始数据长度：{len(raw_signal)} 采样点")
    print(f"前10个采样值：{raw_signal[:10]}")
    
    # 执行完整分析
    processor = ECGProcessor()
    success, results, _ = processor.analyze_ecg_file(filepath)
    
    if success:
        print("\n===== 分析成功 =====")
        print(f"心率：{results.get('heart_rate', 'N/A')} BPM")
        print(f"R峰数量：{len(results['wave_features'].get('r_peaks', []))}")
        print(f"HRV-RMSSD：{results.get('hrv_analysis', {}).get('rmssd', 'N/A')} ms")
    else:
        print("\n===== 分析失败 =====")
        print(f"错误信息：{results.get('error', '未知错误')}")

if __name__ == "__main__":
    debug_analysis("/app/data/test.dat")  # 容器内路径