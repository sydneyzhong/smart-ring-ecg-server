import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
import json
from datetime import datetime
from collections import defaultdict
from matplotlib.font_manager import FontProperties

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============ 新增的HRV分析导入（放在所有导入之后，类定义之前） ============
try:
    from hrvanalysis import (
        get_time_domain_features,
        get_frequency_domain_features,
        get_csi_cvi_features  # 非线性分析
    )
except ImportError as e:
    print(f"警告: hrv-analysis未正确安装({str(e)})，使用模拟模式")
    
    # 模拟函数定义
    def get_time_domain_features(rr_intervals):
        return {
            "rmssd": 28.6, 
            "sdnn": 45.2,
            "mean_hr": 72.0,
            "pnni_50": 12.3
        }
    
    def get_frequency_domain_features(rr_intervals):
        return {
            "lf": 456.7,
            "hf": 321.5,
            "lf_hf_ratio": 1.42
        }
    
    def get_csi_cvi_features(rr_intervals):
        return {
            "csi": 25.8,
            "cvi": 32.1
        }
# ============ 新增内容结束 ============

OUTPUT_DIR = "/app/reports"  # 必须与docker-compose中的挂载目录一致
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保目录存在

class ECGProcessor:
    def __init__(self, fs=250):
        import matplotlib
        matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'  # 指定中文字体
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        self.sample_rate = fs
        # ...其他初始化代码...

        self.fs = fs
        self._set_chinese_font()
        self.healthy_ranges = {
            'hr': (60, 100),
            'hrv_rmssd': (20, 60),
            'qrs_width': (80, 120),
            'pr_interval': (120, 200),
            'qt_interval': (350, 440),
            'qtc': (340, 450)
        }

        # 完整的24种疾病特征库（替换原有的arrhythmia_library）
        self.disease_library = {
            # 一、心律失常类（8种）
            '心房扑动': {
                'type': '心律失常',
                'features': {
                    'hr': (250, 350),
                    'f_waves': True,
                    'regularity': '规则'
                },
                'risk_level': '中高',
                'description': '心房快速规则活动，心室率通常规则'
            },
            '心房颤动': {
                'type': '心律失常',
                'features': {
                    'irregular': True,
                    'p_waves': '缺失',
                    'fibrillatory_waves': True
                },
                'risk_level': '高',
                'description': '心房电活动紊乱，心室率绝对不齐'
            },
            '室性早搏': {
                'type': '心律失常',
                'features': {
                    'qrs_width': '>120ms',
                    'compensatory_pause': '完全',
                    'p_waves': '无关'
                },
                'risk_level': '中',
                'description': '心室提前除极引起的异常搏动'
            },
            '室性心动过速': {
                'type': '心律失常',
                'features': {
                    'hr': '>100',
                    'qrs_width': '>120ms',
                    'consecutive': '>=3'
                },
                'risk_level': '极高',
                'description': '连续3个以上室性早搏'
            },
            '房室传导阻滞（一度）': {
                'type': '传导异常',
                'features': {
                    'pr_interval': '>200ms',
                    'qrs_width': '<120ms'
                },
                'risk_level': '中',
                'description': 'PR间期延长但每个P波都能下传'
            },
            '房室传导阻滞（二度I型）': {
                'type': '传导异常',
                'features': {
                    'pr_prolongation': True,
                    'dropped_beats': True
                },
                'risk_level': '中高',
                'description': 'PR间期逐渐延长直至QRS脱落'
            },
            '预激综合征（WPW）': {
                'type': '传导异常',
                'features': {
                    'pr_interval': '<120ms',
                    'delta_wave': True,
                    'qrs_width': '>110ms'
                },
                'risk_level': '中高',
                'description': '存在房室旁路导致心室预激'
            },
            '交界性心律': {
                'type': '心律失常',
                'features': {
                    'qrs_width': '<120ms',
                    'p_waves': '逆行或无'
                },
                'risk_level': '低',
                'description': '房室交界区发出的心律'
            },
            
            # 二、心肌缺血/梗死类（3种）
            '心肌缺血': {
                'type': '心肌异常',
                'features': {
                    'st_segment': ('压低', '水平'),
                    't_waves': ('倒置', '平坦'),
                    'duration': '>1min'
                },
                'risk_level': '中高',
                'description': '心内膜下心肌供血不足'
            },
            '急性心肌梗死': {
                'type': '心肌梗死',
                'features': {
                    'st_segment': '抬高',
                    'q_waves': '病理性',
                    't_waves': '动态演变'
                },
                'risk_level': '极高',
                'description': '冠状动脉急性闭塞导致心肌坏死'
            },
            '心内膜下缺血': {
                'type': '心肌异常',
                'features': {
                    't_waves': '深倒置',
                    'st_segment': '轻度压低'
                },
                'risk_level': '中',
                'description': '广泛心内膜下缺血'
            },
            
            # 三、电解质/代谢类（3种）
            '低钾血症': {
                'type': '电解质紊乱',
                'features': {
                    'u_waves': '增高',
                    't_waves': '低平',
                    'st_segment': '压低'
                },
                'risk_level': '中',
                'description': '血清钾浓度＜3.5mmol/L'
            },
            '高钾血症': {
                'type': '电解质紊乱',
                'features': {
                    't_waves': '高尖',
                    'qrs_width': '增宽',
                    'p_waves': '减小'
                },
                'risk_level': '高',
                'description': '血清钾浓度＞5.5mmol/L'
            },
            '洋地黄效应': {
                'type': '药物影响',
                'features': {
                    'st_segment': '下斜型压低',
                    't_waves': '鱼钩样'
                },
                'risk_level': '中',
                'description': '洋地黄类药物导致的特征性改变'
            },
            
            # 四、遗传/原发性（3种）
            '长QT综合征': {
                'type': '遗传性',
                'features': {
                    'qtc': '>450ms',
                    't_waves': ('切迹', '交替'),
                    'torsades': '可能'
                },
                'risk_level': '高',
                'description': '心肌复极延长导致的恶性心律失常风险'
            },
            'Brugada综合征': {
                'type': '遗传性',
                'features': {
                    'st_segment': '马鞍形',
                    'leads': ('V1', 'V2'),
                    'hr': '正常'
                },
                'risk_level': '极高',
                'description': '钠离子通道异常导致的猝死高风险'
            },
            '早复极综合征': {
                'type': '原发性',
                'features': {
                    'j_point': '抬高',
                    'st_segment': '凹面向上'
                },
                'risk_level': '低',
                'description': '良性J点抬高现象'
            },
            
            # 五、其他全身性（7种）
            '肺栓塞': {
                'type': '肺源性',
                'features': {
                    'pattern': 'S1Q3T3',
                    'sinus_tachycardia': True,
                    't_waves': '倒置'
                },
                'risk_level': '高',
                'description': '肺动脉血栓导致右心负荷增加'
            },
            '颅内压增高': {
                'type': '神经系统',
                'features': {
                    't_waves': '深倒置',
                    'qt_interval': '延长'
                },
                'risk_level': '高',
                'description': '脑部病变导致的特征性改变'
            },
            '甲状腺功能亢进': {
                'type': '内分泌',
                'features': {
                    'hr': '>100',
                    'st_t_changes': '非特异性'
                },
                'risrisk_levelk': '中',
                'description': '甲状腺激素过多导致的心动过速'
            },
            '迷走神经张力过高': {
                'type': '自主神经',
                'features': {
                    'hr': '<60',
                    'respiratory_variation': True
                },
                'risrisk_levelk': '低',
                'description': '迷走神经优势导致的心动过缓'
            },
            '体位性心动过速': {
                'type': '自主神经',
                'features': {
                    'hr_increase': '>30bpm',
                    'postural_change': True
                },
                'ririsk_levelsk': '中',
                'description': '体位改变时心率异常增加'
            },
            '睡眠呼吸暂停': {
                'type': '呼吸性',
                'features': {
                    'hr_variation': '周期性',
                    'bradycardia': '夜间',
                    'qt_interval': '延长'
                },
                'ririsk_levelsk': '中',
                'description': '睡眠期间反复呼吸暂停导致缺氧'
            },
            '慢性阻塞性肺病': {
                'type': '呼吸性',
                'features': {
                    'p_pulmonale': True,
                    'right_axis_deviation': True
                },
                'riskrisk_level': '中',
                'description': '慢性肺病导致的右心负荷增加'
            }
        }

    def _calculate_heart_rate(self, r_peaks, fs):
        """基于有效R峰计算平均心率"""
        if len(r_peaks) < 2:
            return 0  # 无法计算
            
        rr_intervals = np.diff(r_peaks) / fs  # 单位：秒
        avg_rr = np.median(rr_intervals)  # 使用中位数抗干扰
        
        # 过滤异常间期（排除<0.3s或>2.0s）
        valid_rr = [rr for rr in rr_intervals if 0.3 < rr < 2.0]
        if len(valid_rr) < 1:
            return 0
            
        heart_rate = 60 / np.mean(valid_rr)
        print(f"[DEBUG] 心率计算中间值：平均RR={np.mean(valid_rr):.2f}s, 心率={heart_rate:.1f} BPM")  # 调试输出
        return round(heart_rate)

    def _set_chinese_font(self):
        """设置中文字体"""
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass


    def _assess_disease_risks(self, results):
        """
        综合评估24种疾病风险
        参数:
            results - 包含完整分析结果的字典
        返回:
            字典格式: { 疾病名称: { 'score': 风险评分, 'risk': 风险等级, 'features': [阳性特征列表] } }
        """
        features = results['wave_features']
        disease_risks = {}

        # 1. 动态计算QTc（Bazett公式）
        qtc = self._calculate_qtc(features.get('qt_interval', 400), 
                                features.get('hr', 60))
        features['qtc'] = qtc

        # 2. 按疾病类别评估
        # 心律失常类评估
        self._assess_arrhythmias(features, disease_risks)
        
        # 心肌缺血/梗死评估
        self._assess_ischemia(features, disease_risks)
        
        # 电解质/代谢异常评估
        self._assess_electrolytes(features, disease_risks)
        
        # 遗传/原发性疾病评估
        self._assess_genetic_disorders(features, disease_risks)
        
        # 全身性疾病评估
        self._assess_systemic_diseases(features, disease_risks)

        return disease_risks

    def _assess_arrhythmias(self, features, risks):
        """评估8种心律失常"""
        # 心房扑动/颤动
        if features.get('f_waves'):
            risks['心房扑动'] = self._calculate_risk('心房扑动', features, {
                'hr': features.get('hr', 0),
                'f_waves': True
            })
        elif features.get('irregular') and not features.get('p_waves'):
            risks['心房颤动'] = self._calculate_risk('心房颤动', features, {
                'hr': features.get('hr', 0)
            })

        # 室性心律失常
        if features['qrs_complex']['width_status'] == '增宽':
            risks['室性早搏'] = self._calculate_risk('室性早搏', features, {
                'qrs_width': features['qrs_complex']['average_width']
            })
            
            # 检测连续室早
            if self._detect_consecutive_wide_qrs(features):
                risks['室性心动过速'] = self._calculate_risk('室性心动过速', features, {
                    'hr': features.get('hr', 0)
                })

        # 传导阻滞
        if features.get('pr_interval', 0) > 200:
            risks['房室传导阻滞（一度）'] = self._calculate_risk('房室传导阻滞（一度）', features, {
                'pr_interval': features['pr_interval']
            })

    def _assess_ischemia(self, features, risks):
        """评估心肌缺血/梗死"""
        st_status = features.get('st_segment', {}).get('status')
        if st_status == '抬高':
            risks['急性心肌梗死'] = self._calculate_risk('急性心肌梗死', features, {
                'st_elevation': features['st_segment']['average_elevation']
            })
        elif st_status == '压低':
            risks['心肌缺血'] = self._calculate_risk('心肌缺血', features, {
                'st_depression': features['st_segment']['average_elevation']
            })
            
        # 心内膜下缺血（广泛T波倒置）
        if features.get('t_waves', {}).get('inverted', False):
            risks['心内膜下缺血'] = self._calculate_risk('心内膜下缺血', features, {
                't_inversion': True
            })

    def _assess_electrolytes(self, features, risks):
        """评估电解质紊乱"""
        # 低钾血症（U波增高）
        if features.get('u_wave_present'):
            risks['低钾血症'] = self._calculate_risk('低钾血症', features, {
                'u_wave_ratio': features.get('u_wave_ratio', 0)
            })
            
        # 高钾血症（T波高尖+QRS增宽）
        if (features.get('t_waves', {}).get('peaked', False) and 
            features['qrs_complex']['average_width'] > 120):
            risks['高钾血症'] = self._calculate_risk('高钾血症', features, {
                'qrs_width': features['qrs_complex']['average_width']
            })

    def _assess_genetic_disorders(self, features, risks):
        """评估遗传性疾病"""
        # 长QT综合征
        if features.get('qtc', 0) > 450:
            risks['长QT综合征'] = self._calculate_risk('长QT综合征', features, {
                'qtc': features['qtc']
            })
            
        # Brugada模式（需要模拟V1-V2导联）
        if self._detect_brugada_pattern(features):
            risks['Brugada综合征'] = self._calculate_risk('Brugada综合征', features)

    def _assess_systemic_diseases(self, features, risks):
        """评估全身性疾病"""
        # 睡眠呼吸暂停（周期性心率变化）
        if self._detect_cyclic_hr_variation(features):
            risks['睡眠呼吸暂停'] = self._calculate_risk('睡眠呼吸暂停', features)
            
        # 肺栓塞（S1Q3T3模式）
        if self._detect_s1q3t3_pattern(features):
            risks['肺栓塞'] = self._calculate_risk('肺栓塞', features)

    def _calculate_risk(self, disease, features, specific_features=None):
        """
        计算单个疾病风险评分
        参数:
            disease - 疾病名称
            features - 全部波形特征
            specific_features - 该疾病特有的特征值
        返回:
            风险字典 {score, risk, features}
        """
        disease_data = self.disease_library.get(disease, {})
        if not disease_data:
            return None
            
        # 1. 基础分（根据风险等级）
        risk_levels = {'极低':1, '低':2, '中':4, '中高':6, '高':8, '极高':10}
        score = risk_levels.get(disease_data['risk_level'], 0)
        
        # 2. 特征匹配加分
        matched_features = []
        for feat, condition in disease_data['features'].items():
            feat_value = specific_features.get(feat, None) if specific_features else features.get(feat, None)
            
            if feat_value is not None:
                if isinstance(condition, tuple):  # 多条件
                    if any(self._match_condition(feat_value, c) for c in condition):
                        score += 2
                        matched_features.append(f"{feat}={feat_value}")
                elif self._match_condition(feat_value, condition):
                    score += 2
                    matched_features.append(f"{feat}={feat_value}")
        
        # 3. 限制分数范围
        score = min(10, max(0, score))
        
        return {
            'score': round(score, 1),
            'risk_level': disease_data['risk_level'],     ### risk--> risk_level
            'description': disease_data['description'],
            'features': matched_features
        }

    def _match_condition(self, value, condition):
        """匹配特征值与条件"""
        if isinstance(condition, str):
            if condition.startswith('>'):
                return value > float(condition[1:])
            elif condition.startswith('<'):
                return value < float(condition[1:])
            else:
                return value == condition
        else:
            return value == condition

    # 新增辅助检测方法
    def _detect_consecutive_wide_qrs(self, features):
        """检测连续宽QRS波（室速）"""
        # 实现逻辑...
        return False

    def _detect_brugada_pattern(self, features):
        """检测Brugada波模式"""
        # 实现逻辑...
        return False

    def _detect_cyclic_hr_variation(self, features):
        """检测周期性心率变化"""
        # 实现逻辑...
        return False

    def _detect_s1q3t3_pattern(self, features):
        """检测S1Q3T3模式"""
        # 实现逻辑...
        return False

    def _calculate_qtc(self, qt, hr):
        """计算校正QT间期（Bazett公式）"""
        rr_interval = 60 / hr if hr > 0 else 1
        return qt / np.sqrt(rr_interval)
 

    def _set_chinese_font(self):
        """统一设置中文字体"""
        try:
            # 尝试多种中文字体方案
            font_path = self._find_chinese_font()
            if font_path:
                font_prop = FontProperties(fname=font_path)
                plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            else:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
            
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"字体设置警告: {str(e)}")

    def _find_chinese_font(self):
        """查找系统中可用的中文字体文件"""
        font_paths = [
            # Windows
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            # Mac
            '/System/Library/Fonts/PingFang.ttc',
            # Linux
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                return path
        return None

    def _generate_html_content(self, analysis, report_data):
        """生成HTML内容（中文版）"""
        # 基础信息
        basic_info = analysis["basic_info"]
        health_index = analysis["health_index"]
        
        # 健康状态
        health_status = "优秀" if health_index >= 80 else "良好" if health_index >= 60 else "需关注"
        health_color = "#27ae60" if health_index >= 80 else "#f39c12" if health_index >= 60 else "#e74c3c"
        
        # 核心指标
        duration = basic_info["duration"]
        qrs_count = analysis["wave_features"]["qrs_complex"]["count"]
        heart_rate = f"{(qrs_count / duration * 60):.0f}" if duration > 0 else "无法计算"
        
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>心电分析报告 - {basic_info['filename']}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', 'PingFang SC', 'SimHei', sans-serif;
            line-height: 1.6;
            color: #333;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .health-score {{
            font-size: 2.5em;
            color: {health_color};
            margin: 10px 0;
        }}
        .chart {{
            width: 100%;
            margin: 15px 0;
            border: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>心电分析报告</h1>
        <div class="health-score">{health_index}/100</div>
        <p>报告生成时间: {basic_info['timestamp']}</p>
    </div>

    <!-- 健康雷达图 -->
    <h2>健康状态雷达图</h2>
    <img src="{os.path.basename(report_data['plots']['health_radar'])}" class="chart">

    <!-- 疾病风险评估 -->
    <h2>疾病风险评估</h2>
    <img src="{os.path.basename(report_data['plots']['disease_risk'])}" class="chart">

    <!-- ECG波形图 -->
    <h2>ECG波形分析</h2>
    <img src="{os.path.basename(report_data['plots']['ecg_waveform'])}" class="chart">

    <!-- 核心指标 -->
    <h2>核心指标</h2>
    <ul>
        <li>平均心率: {heart_rate} 次/分钟</li>
        <li>健康指数: {health_index}/100 ({health_status})</li>
    </ul>

    <!-- 免责声明 -->
    <div style="margin-top: 30px; color: #777; font-size: 0.9em;">
        <p>注：本报告仅供参考，不能作为医疗诊断依据</p>
    </div>
</body>
</html>"""

    def _generate_placeholder_image(self, save_path, title):
        """生成占位图（中文）"""
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f"{title}\n(图表未生成)", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        return True

    # ... (其他方法保持不变)
    def _get_basic_info(self, ecg_signal, filename):
        """获取ECG基础信息"""
        return {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "filename": filename,
            "duration": len(ecg_signal) / self.fs,  # 使用实例变量self.fs
            "samples": len(ecg_signal),
            "fs": self.fs,  # 新增采样率字段
            "ecg_signal": ecg_signal.tolist()
        }

  
    def analyze_ecg_file(self, filepath):
        """分析ECG文件主方法（修复版）"""
        try:
            # 1. 验证文件
            if not os.path.exists(filepath):
                return False, {"error": "文件不存在"}, None
            if os.path.getsize(filepath) == 0:
                return False, {"error": "空文件"}, None

            # 2. 读取数据
            ecg_signal = np.fromfile(filepath, dtype=np.int16)
            filename = os.path.basename(filepath)
            
            # 3. 创建报告目录
            report_dir = "/app/reports"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"{filename.split('.')[0]}_report.html")
            
            # 4. 执行分析
            results = {
                "basic_info": self._get_basic_info(ecg_signal, filename),
                "wave_features": {},
                "health_index": 0
            }

            # 添加心率计算
            duration = results["basic_info"]["duration"]
            qrs_count = len(r_peaks)
            results["heart_rate"] = self._calculate_heart_rate(qrs_count, duration) 

            r_peaks = self._detect_r_peaks(ecg_signal)
            results["wave_features"] = {
                "r_peaks": r_peaks.tolist(),
                **self._analyze_qrs_complex(ecg_signal, r_peaks),
                **self._analyze_pt_waves(ecg_signal, r_peaks)
            }
            
            if len(r_peaks) >= 2:
                results.update({
                    "hrv_analysis": self._analyze_hrv(r_peaks),
                    "arrhythmia": self._check_arrhythmia(r_peaks),
                    "disease_risks": self._assess_disease_risks(results)
                })
            
            results["health_index"] = self._calculate_health_index(results)
            
            # 5. 生成报告
            self._generate_html_report(results, report_path)
            
            return True, results, {"html_report": report_path}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}, None
        

    def _analyze_st_segment(self, ecg, r_peaks):
        """ST段分析（新增）"""
        if len(r_peaks) < 2:
            return {"st_segment": {"status": "未检测到", "average_elevation": 0}}
        
        st_segments = []
        for peak in r_peaks:
            j_point = min(len(ecg), peak + int(0.08*self.fs))  # J点
            st_end = min(len(ecg), peak + int(0.16*self.fs))   # ST段终点
            
            if st_end >= len(ecg):
                continue
                
            baseline = np.mean(ecg[max(0, peak-100):peak-20])  # TP段作为基线
            st_level = np.mean(ecg[j_point:st_end]) - baseline
            st_segments.append(st_level)
        
        if not st_segments:
            return {"st_segment": {"status": "未检测到", "average_elevation": 0}}
        
        avg_st = np.mean(st_segments)
        st_status = "正常" if -0.05 <= avg_st <= 0.1 else ("抬高" if avg_st > 0.1 else "压低")
        
        return {
            "st_segment": {
                "status": st_status,
                "average_elevation": float(avg_st),
                "assessment": f"ST段{st_status} ({avg_st:.2f}mV)"
            }
        }


    def _bandpass_filter(self):
        # 实现带通滤波逻辑（例如使用 scipy.signal）
        from scipy.signal import butter, filtfilt
        nyquist = 0.5 * self.sample_rate
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, self.signal)
        return filtered

    def _detect_r_peaks(self, ecg_signal):
        """增强鲁棒性的R波检测"""
        try:
            filtered_signal = self._bandpass_filter()  # 确保调用正确
            # 带通滤波参数调整
            filtered = self._bandpass_filter(ecg_signal, lowcut=8.0, highcut=15.0)
            
            # 动态阈值计算（基于统计分布）
            mean_val = np.mean(filtered)
            std_val = np.std(filtered)
            height_threshold = mean_val + 4 * std_val
            
            # 主检测
            peaks, _ = find_peaks(
                filtered,
                height=height_threshold,
                distance=int(0.3 * self.fs),  # 允许更密集的R波
                prominence=std_val * 0.8,
                width=(int(0.04 * self.fs), int(0.12 * self.fs))
            )
            print(f"[DEBUG] 初步检测到 {len(peaks)} 个R峰")  # 调试输出
            
            # 二次验证（基于相邻RR间期）
            valid_peaks = []
            prev_peak = -np.inf
            for peak in peaks:
                if (peak - prev_peak) > 0.2 * self.fs:  # 最小间隔200ms
                    valid_peaks.append(peak)
                    prev_peak = peak
                    
            print(f"[DEBUG] 最终有效R峰数量：{len(valid_peaks)}")  # 调试输出
            return np.array(valid_peaks)
            
        except Exception as e:
            print(f"[ERROR] R峰检测失败: {str(e)}")
            return np.array([])

    def _analyze_qrs_complex(self, ecg, r_peaks):
        """QRS波群分析"""
        if len(r_peaks) == 0:
            return {
                "qrs_complex": {
                    "count": 0,
                    "average_width": 0,
                    "width_status": "未检测到",
                    "amplitude": 0
                }
            }
        
        qrs_list = []
        for peak in r_peaks:
            q_start = max(0, peak - int(0.05*self.fs))
            s_end = min(len(ecg), peak + int(0.05*self.fs))
            qrs_list.append({
                "position": int(peak),
                "width": (s_end - q_start)/self.fs * 1000,
                "amplitude": float(ecg[peak])
            })
        
        avg_width = np.mean([q["width"] for q in qrs_list])
        return {
            "qrs_complex": {
                "count": len(r_peaks),
                "average_width": float(avg_width),
                "width_status": self._assess_parameter(avg_width, 'qrs_width'),
                "amplitude": float(np.mean([q["amplitude"] for q in qrs_list]))
            }
        }

    def _analyze_pt_waves(self, ecg, r_peaks):
        """P波和T波分析"""
        def analyze_wave(r_positions, window_start, window_end):
            waves = []
            for r_pos in r_positions:
                start = max(0, r_pos + int(window_start*self.fs))
                end = min(len(ecg), r_pos + int(window_end*self.fs))
                if start >= end:
                    continue
                    
                segment = ecg[start:end]
                if len(segment) == 0:
                    continue
                    
                pos = start + np.argmax(segment)
                waves.append({
                    "position": int(pos),
                    "amplitude": float(ecg[pos]),
                    "interval": float((pos - r_pos)/self.fs * 1000)
                })
            return waves

        p_waves = analyze_wave(r_peaks, -0.2, -0.12)  # P波
        t_waves = analyze_wave(r_peaks, 0.2, 0.4)     # T波

        return {
            "p_waves": self._summarize_waves(p_waves, 'P'),
            "t_waves": self._summarize_waves(t_waves, 'T')
        }

    def _summarize_waves(self, wave_list, wave_type):
        """波形特征汇总"""
        if not wave_list:
            return {
                "detected": False,
                "assessment": f"未检测到{wave_type}波",
                "average_amplitude": 0,
                f"average_{'pr' if wave_type=='P' else 'qt'}_interval": 0
            }
        
        intervals = [w["interval"] for w in wave_list]
        avg_interval = np.mean(intervals)
        
        return {
            "detected": True,
            "count": len(wave_list),
            "average_amplitude": float(np.mean([w["amplitude"] for w in wave_list])),
            f"average_{'pr' if wave_type=='P' else 'qt'}_interval": float(avg_interval),
            "assessment": self._assess_wave(wave_type, 
                                          np.mean([w["amplitude"] for w in wave_list]),
                                          avg_interval),
            "details": wave_list
        }

    def _analyze_hrv(self, r_peaks):
        """增强版HRV计算"""
        if len(r_peaks) < 2:
            return {"rmssd": 0, "sdnn": 0, "assessment": "数据不足"}
        
        try:
            rr_intervals = np.diff(r_peaks) / self.fs * 1000  # 转换为毫秒
            
            # 异常值过滤（排除差异>30%的间隔）
            median_rr = np.median(rr_intervals)
            filtered_rr = [rr for rr in rr_intervals if 0.7*median_rr < rr < 1.3*median_rr]
            
            if len(filtered_rr) < 2:
                return {"rmssd": 0, "sdnn": 0, "assessment": "有效数据不足"}
                
            rmssd = np.sqrt(np.mean(np.square(np.diff(filtered_rr))))
            return {
                "rmssd": float(rmssd),
                "sdnn": float(np.std(filtered_rr)),
                "assessment": self._assess_hrv(rmssd)
            }
        except Exception as e:
            print(f"[ERROR] HRV计算失败: {str(e)}")
            return {"rmssd": 0, "sdnn": 0, "assessment": "计算错误"}
            

    def _check_arrhythmia(self, r_peaks):
        """心律失常检测"""
        rr_intervals = np.diff(r_peaks) / self.fs * 1000
        rr_variation = np.std(rr_intervals) / np.mean(rr_intervals)
        
        arrhythmia_types = []
        if rr_variation > 0.25:
            arrhythmia_types.append("心律不齐")
        if any(rr > 1200 for rr in rr_intervals):
            arrhythmia_types.append("心动过缓")
        if any(rr < 600 for rr in rr_intervals):
            arrhythmia_types.append("心动过速")
        
        return {
            "types": arrhythmia_types,
            "conclusion": "正常心律" if not arrhythmia_types else "，".join(arrhythmia_types)
        }

    def _calculate_health_index(self, results):
        """计算心脏健康指数"""
        try:
            score = 70  # 基础分
            
            # HRV评分
            if "hrv_analysis" in results and "rmssd" in results["hrv_analysis"]:
                rmssd = results["hrv_analysis"]["rmssd"]
                if rmssd > 50: score += 10
                elif rmssd > 30: score += 5
            
            # QRS评分
            if "qrs_complex" in results["wave_features"]:
                qrs_width = results["wave_features"]["qrs_complex"]["average_width"]
                if 80 <= qrs_width <= 120: score += 5
            
            # 心律失常扣分
            if "types" in results["arrhythmia"]:
                score -= len(results["arrhythmia"]["types"]) * 10
            
            return max(0, min(100, int(score)))
        except:
            return 0

    def _assess_parameter(self, value, param_type):
        """评估单个参数"""
        low, high = self.healthy_ranges[param_type]
        if value < low:
            return "偏低"
        elif value > high:
            return "偏高"
        return "正常"

    def _assess_hrv(self, rmssd):
        """评估HRV状态"""
        if rmssd < 20:
            return "自主神经功能失衡"
        elif rmssd < 30:
            return "轻度压力"
        else:
            return "自主神经功能平衡"

    def _assess_wave(self, wave_type, amplitude, interval):
        """波形健康评估"""
        thresholds = {
            "P": {"amp": (0.05, 0.25), "interval": (120, 200)},
            "T": {"amp": (0.1, 0.5), "interval": (160, 350)}
        }
        
        amp_ok = thresholds[wave_type]["amp"][0] <= amplitude <= thresholds[wave_type]["amp"][1]
        int_ok = thresholds[wave_type]["interval"][0] <= interval <= thresholds[wave_type]["interval"][1]
        
        if amp_ok and int_ok:
            return f"{wave_type}波形态正常"
        elif not amp_ok and not int_ok:
            return f"{wave_type}波振幅和间期异常"
        elif not amp_ok:
            return f"{wave_type}波振幅异常"
        else:
            return f"{wave_type}波间期异常"

    def generate_report(self, results, filename):
        """生成可视化报告（增强版）"""
        report = {
            "status": "success",
            "text_report": "",
            "plots": {},
            "json_path": "",
            "html_report": None
        }
        
        try:
            # 创建输出目录
            base_name = os.path.splitext(filename)[0]
            file_output_dir = os.path.join(OUTPUT_DIR, base_name)
            os.makedirs(file_output_dir, exist_ok=True)
            
            # 生成文字报告
            report["text_report"] = self._generate_text_report(results)
            with open(os.path.join(file_output_dir, "report.txt"), 'w', encoding='utf-8') as f:
                f.write(report["text_report"])
            
            # 生成图表
            if "ecg_signal" in results["basic_info"]:
                ecg_signal = np.array(results["basic_info"]["ecg_signal"])
                
                # ECG波形图
                ecg_plot_path = os.path.join(file_output_dir, "ecg_waveform.png")
                if len(results["wave_features"].get("r_peaks", [])) > 0:
                    self._plot_ecg_waveform(ecg_signal, 
                                        results["wave_features"]["r_peaks"],
                                        results["wave_features"].get("p_waves", {}),
                                        results["wave_features"].get("t_waves", {}),
                                        ecg_plot_path)
                    report["plots"]["ecg_waveform"] = ecg_plot_path
                
                # 健康雷达图
                radar_path = os.path.join(file_output_dir, "health_radar.png")
                if self._plot_health_radar(results, radar_path):
                    report["plots"]["health_radar"] = radar_path
                
                # 疾病风险图（新增）
                risk_path = os.path.join(file_output_dir, "disease_risk.png")
                if self._plot_disease_risk(results, risk_path):
                    report["plots"]["disease_risk"] = risk_path
            
            # 保存JSON结果
            report["json_path"] = os.path.join(file_output_dir, "analysis_results.json")
            with open(report["json_path"], 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 生成HTML报告
            if report["plots"].get("health_radar"):
                report["html_report"] = self.generate_html_report({
                    "json_path": report["json_path"],
                    "plots": report["plots"]
                }, file_output_dir)
                
        except Exception as e:
            report["status"] = "error"
            report["message"] = f"生成报告时出错: {str(e)}"
            import traceback
            traceback.print_exc()
        
        return report
    


    def _plot_disease_risk(self, results, save_path):
        """绘制疾病风险图（完整中文支持版）"""
        try:
            if not results.get("disease_risks"):
                return False  # 没有风险数据时不生成图表

            # 准备数据
            diseases = []
            scores = []
            colors = []
            risk_colors = {
                '极低': '#2ecc71',
                '低': '#3498db',
                '中': '#f39c12',
                '中高': '#e74c3c',
                '高': '#c0392b',
                '极高': '#9b59b6'
            }

            for disease, data in results["disease_risks"].items():
                diseases.append(disease)
                scores.append(data["score"])
                colors.append(risk_colors.get(data["risk_level"], '#95a5a6'))

            # 设置中文字体（三种方案确保兼容）
            try:
                # 方案1：使用系统已安装的字体文件
                font_path = self._find_chinese_font()
                if font_path:
                    zh_font = FontProperties(fname=font_path)
                    plt.rcParams['font.sans-serif'] = [zh_font.get_name()]
                else:
                    # 方案2：使用常见中文字体名称
                    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
                
                # 通用设置
                plt.rcParams['axes.unicode_minus'] = False
            except Exception as font_error:
                print(f"字体设置警告: {str(font_error)}")

            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(diseases))
            bars = ax.barh(y_pos, scores, color=colors, alpha=0.7)

            # 设置中文标签（强制指定字体属性）
            try:
                if 'zh_font' in locals():
                    # 使用找到的字体文件
                    ax.set_yticklabels(diseases, fontproperties=zh_font, fontsize=10)
                    ax.set_xlabel('风险评分 (0-10)', fontproperties=zh_font, fontsize=10)
                    ax.set_title('心脏疾病风险评估', fontproperties=zh_font, fontsize=12)
                else:
                    # 回退到系统字体
                    ax.set_yticklabels(diseases, fontsize=10)
                    ax.set_xlabel('风险评分 (0-10)', fontsize=10)
                    ax.set_title('心脏疾病风险评估', fontsize=12)
            except Exception as label_error:
                print(f"标签设置警告: {str(label_error)}")
                # 终极回退方案：英文显示
                ax.set_xlabel('Risk Score (0-10)', fontsize=10)
                ax.set_title('Disease Risk Assessment', fontsize=12)

            ax.set_yticks(y_pos)
            ax.set_xlim(0, 10)

            # 添加数据标签
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}', ha='left', va='center', fontsize=8)

            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
            return True

        except Exception as e:
            print(f"疾病风险图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _find_chinese_font(self):
        """查找系统中可用的中文字体文件"""
        font_paths = [
            # Windows
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            # Mac
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            # Linux
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                return path
        
        # 尝试通过matplotlib查找
        try:
            from matplotlib import font_manager
            for font in font_manager.fontManager.ttflist:
                if any(name.lower() in font.name.lower() for name in ['YaHei', 'Hei', 'PingFang', 'STSong']):
                    return font.fname
        except:
            pass
        
        return None




    def _plot_ecg_waveform(self, ecg, r_peaks, p_waves, t_waves, save_path):
        """绘制ECG波形图"""
        plt.figure(figsize=(12, 4))
        t = np.arange(len(ecg)) / self.fs
        
        # 绘制ECG信号
        plt.plot(t, ecg, '#1f77b4', linewidth=1, alpha=0.8)
        
        # 标记特征点
        plt.plot(t[r_peaks], ecg[r_peaks], 'ro', markersize=4, label='R峰')
        
        if p_waves.get("detected", False):
            p_pos = [w["position"] for w in p_waves.get("details", [])]
            plt.plot(t[p_pos], ecg[p_pos], 'g^', markersize=4, label='P波')
        
        if t_waves.get("detected", False):
            t_pos = [w["position"] for w in t_waves.get("details", [])]
            plt.plot(t[t_pos], ecg[t_pos], 'mv', markersize=4, label='T波')
        
        plt.title("ECG波形分析 - " + os.path.basename(save_path).split('.')[0], fontsize=12)
        plt.xlabel("时间 (秒)", fontsize=10)
        plt.ylabel("振幅 (mV)", fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()

    def _plot_health_radar(self, results, save_path):
        """生成健康指数雷达图（完整修复版）"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import FontProperties
            import matplotlib
            
            # 1. 准备雷达图数据
            categories = ['心率', '变异性', 'QRS波', '心律', '综合']
            values = [
                min(100, results["wave_features"].get("qrs_complex", {}).get("count", 0)),
                results.get("hrv_analysis", {}).get("rmssd", 0) * 2,
                100 - abs(results["wave_features"].get("qrs_complex", {}).get("average_width", 100) - 100),
                100 - len(results.get("arrhythmia", {}).get("types", [])) * 25,
                results.get("health_index", 0)
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))
            
            # 2. 设置中文字体（兼容方案）
            font_path = None
            font_options = [
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # Linux
                'C:/Windows/Fonts/msyh.ttc',  # Windows
                '/System/Library/Fonts/PingFang.ttc',  # Mac
                '/usr/share/fonts/truetype/arphic/uming.ttc'  # 备用
            ]
            
            for fp in font_options:
                if os.path.exists(fp):
                    font_path = fp
                    break
            
            # 3. 创建图形
            fig = plt.figure(figsize=(8, 8), dpi=120)
            ax = fig.add_subplot(111, polar=True)
            
            # 4. 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
            ax.fill(angles, values, color='#1f77b4', alpha=0.25)
            
            # 5. 设置标签（处理中文显示）
            if font_path:
                zh_font = FontProperties(fname=font_path, size=12)
                ax.set_thetagrids(angles[:-1] * 180/np.pi, categories, fontproperties=zh_font)
                ax.set_title("心脏健康指数", fontproperties=zh_font)
            else:
                # 无中文字体时的降级处理
                ax.set_thetagrids(angles[:-1] * 180/np.pi, ['HR', 'HRV', 'QRS', 'Rhythm', 'Overall'])
                ax.set_title("Heart Health Index")
            
            ax.set_rlim(0, 100)
            ax.grid(True, linestyle=':', alpha=0.5)
            
            # 6. 保存图像
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=120)
            plt.close()
            return True
            
        except Exception as e:
            print(f"❌ 雷达图生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_ecg_file(self, filepath):
        """分析ECG文件主方法"""
        try:
            # 读取数据时添加字节顺序和大端模式支持
            ecg_signal = np.fromfile(filepath, dtype='>i2')  # 假设数据是大端格式
            
            # 数据完整性检查
            if len(ecg_signal) < 250 * 10:  # 至少10秒数据（假设采样率250Hz）
                return False, {"error": "数据过短（需至少10秒）"}, None
                
            print(f"[DEBUG] 成功加载信号数据，长度：{len(ecg_signal)} 采样点")
            print(f"[DEBUG] 前10个采样值：{ecg_signal[:10]}")  # 调试输出
            
            # 后续分析逻辑保持不变...

            # 1. 读取并解析数据
            ecg_signal = np.fromfile(filepath, dtype=np.int16)
            filename = os.path.basename(filepath)
            
            # 2. 基础分析
            results = {
                "basic_info": self._get_basic_info(ecg_signal, filename),
                "wave_features": {},
                "health_index": 0
            }
            
            # 3. 特征检测
            r_peaks = self._detect_r_peaks(ecg_signal)
            results["wave_features"] = {
                "r_peaks": r_peaks.tolist(),
                **self._analyze_qrs_complex(ecg_signal, r_peaks),
                **self._analyze_pt_waves(ecg_signal, r_peaks)
            }
            
            # 4. 高级分析（需至少2个R峰）
            if len(r_peaks) >= 2:
                results["hrv_analysis"] = self._analyze_hrv(r_peaks)
                results["arrhythmia"] = self._check_arrhythmia(r_peaks)
                results["disease_risks"] = self._assess_disease_risks(results)
            
            # 5. 健康指数计算
            results["health_index"] = self._calculate_health_index(results)
            
            # 6. 生成可视化报告
            report_path = f"/tmp/reports/{filename.split('.')[0]}_report.html"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            self._generate_html_report(results, report_path)
            
            return True, results, {"html_report": report_path}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}, None


    def _generate_html_report(self, results, output_path):
        """生成HTML格式报告（修复版）"""
        try:
            # 确保报告目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 生成图表路径
            img_dir = os.path.join(os.path.dirname(output_path), "images")
            os.makedirs(img_dir, exist_ok=True)
            ecg_plot_path = os.path.join(img_dir, "ecg_plot.png")
            
            # 绘制ECG信号图
            plt.figure(figsize=(15, 6))
            plt.plot(results["basic_info"]["ecg_signal"][:1000])
            plt.title("ECG Signal Segment")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.savefig(ecg_plot_path)
            plt.close()
            
            # HTML内容
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ECG Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .report-section {{ margin-bottom: 30px; }}
                    img {{ max-width: 100%; }}
                </style>
            </head>
            <body>
                <h1>ECG Analysis Report</h1>
                
                <div class="report-section">
                    <h2>Basic Information</h2>
                    <p>File: {results["basic_info"]["filename"]}</p>
                    <p>Duration: {results["basic_info"]["duration"]:.2f} seconds</p>
                    <p>Sampling Rate: {results["basic_info"].get("fs", 250)} Hz</p>  <!-- 安全访问fs -->
                </div>
                
                <div class="report-section">
                    <h2>ECG Signal</h2>
                    <img src="images/ecg_plot.png" alt="ECG Signal">
                </div>
                
                <div class="report-section">
                    <h2>Analysis Results</h2>
                    <pre>{json.dumps(results, indent=2, ensure_ascii=False)}</pre>
                </div>
            </body>
            </html>
            """
            
            # 写入HTML文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            return True
        except Exception as e:
            print(f"生成报告失败: {str(e)}")
            return False
        

    def _generate_disease_risk_table(self, analysis):
        """生成疾病风险表格（新增）"""
        if not analysis.get("disease_risks"):
            return "<p>未检测到显著疾病风险</p>"
        
        table_rows = []
        risk_level_mapping = {
            '极低': 'low',
            '低': 'low',
            '中': 'medium',
            '中高': 'high',
            '高': 'high',
            '极高': 'veryhigh'
        }
        
        for disease, data in analysis["disease_risks"].items():
            risk_class = f"risk-{risk_level_mapping.get(data['risk_level'], '')}"
            
            table_rows.append(f"""
                <tr>
                    <td>{disease}</td>
                    <td class="{risk_class}">{data['risk_level']}</td>
                    <td>{data['score']:.1f}/10</td>
                    <td>{data['description']}</td>
                </tr>
            """)
        
        return f"""
        <table class="risk-table">
            <thead>
                <tr>
                    <th>疾病名称</th>
                    <th>风险等级</th>
                    <th>风险评分</th>
                    <th>特征描述</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
        """
        

    def _get_detailed_interpretation(self, analysis):
        """生成详细解读内容"""
        hrv_assessment = analysis.get('hrv_analysis', {}).get('assessment', '无法评估')
        p_wave_status = analysis.get('wave_features', {}).get('p_waves', {}).get('assessment', '无法评估')
        t_wave_status = analysis.get('wave_features', {}).get('t_waves', {}).get('assessment', '无法评估')
        arrhythmia_types = analysis.get('arrhythmia', {}).get('types', ['无'])
        st_status = analysis.get('wave_features', {}).get('st_segment', {}).get('assessment', '无法评估')
        
        return f"""
        <div class="interpretation-item">1. 自主神经系统: {hrv_assessment}</div>
        <div class="interpretation-item">2. P波状态: {p_wave_status}</div>
        <div class="interpretation-item">3. T波状态: {t_wave_status}</div>
        <div class="interpretation-item">4. ST段状态: {st_status}</div>
        <div class="interpretation-item">5. 节律异常: {', '.join(arrhythmia_types)}</div>
        """

    def _get_health_color(self, score):
        """根据健康分数返回颜色"""
        if score >= 80:
            return "#27ae60"  # 绿色
        elif score >= 60:
            return "#f39c12"  # 橙色
        else:
            return "#e74c3c"  # 红色

    def _generate_text_report(self, results):
        """生成文字报告"""
        try:
            # 基础信息
            basic_info = results.get("basic_info", {})
            duration = basic_info.get("duration", 0)
            qrs_count = results.get("wave_features", {}).get("qrs_complex", {}).get("count", 0)
            heart_rate = qrs_count / duration * 60 if duration > 0 else 0
            
            # 核心指标
            hrv = results.get("hrv_analysis", {})
            qrs = results.get("wave_features", {}).get("qrs_complex", {})
            arrhythmia = results.get("arrhythmia", {})
            
            # 详细解读
            p_waves = results.get("wave_features", {}).get("p_waves", {})
            t_waves = results.get("wave_features", {}).get("t_waves", {})
            st_segment = results.get("wave_features", {}).get("st_segment", {})

            # 疾病风险评估
            disease_risks = results.get("disease_risks", {})
            top_risks = sorted(disease_risks.items(), key=lambda x: x[1]["score"], reverse=True)[:3]
            
            report = f"""【智能心电分析报告】
⏱ 检测时间: {basic_info.get('timestamp', '未知')}
📄 文件名: {basic_info.get('filename', '未知')}
⏳ 记录时长: {duration:.1f}秒
📊 采样数: {basic_info.get('samples', 0)}
❤️ 平均心率: {heart_rate:.0f} 次/分钟

🔍【核心指标分析】
• 心率变异性(RMSSD): {hrv.get('rmssd', 0):.1f} 毫秒 ({hrv.get('assessment', '无法评估')})
• QRS波宽度: {qrs.get('average_width', 0):.1f} 毫秒 ({qrs.get('width_status', '无法评估')})
• 心律评估: {arrhythmia.get('conclusion', '无法评估')}
• ST段状态: {st_segment.get('assessment', '无法评估')}

💯【心脏健康指数】{results.get('health_index', 0)}/100
{'⭐' * (results.get('health_index', 0) // 20)}

⚠️【主要疾病风险】
{self._format_top_risks(top_risks) if top_risks else "未检测到显著疾病风险"}

📝【详细解读】
1. 自主神经系统: {hrv.get('assessment', '无法评估')}
2. P波状态: {p_waves.get('assessment', '无法评估')}
3. T波状态: {t_waves.get('assessment', '无法评估')}
4. ST段状态: {st_segment.get('assessment', '无法评估')}
5. 节律异常: {', '.join(arrhythmia.get('types', ['无']))}

💡【健康建议】
{self._generate_recommendations(results)}

⚠️ 注：本报告仅供参考，不能替代专业医疗诊断"""
            return report

        except Exception as e:
            return f"生成报告时出错: {str(e)}"

    def _format_top_risks(self, top_risks):
        """格式化主要风险输出（新增）"""
        risk_lines = []
        for disease, data in top_risks:
            risk_lines.append(f"• {disease}: {data['risk_level']}风险 ({data['score']:.1f}/10) - {data['description']}")
        return "\n".join(risk_lines)

    def _generate_recommendations(self, results):
        """生成个性化建议"""
        suggestions = []
        
        try:
            # 基于心率的建议
            duration = results["basic_info"].get("duration", 1)
            hr = results["wave_features"].get("qrs_complex", {}).get("count", 0) / duration * 60
            if hr < 60:
                suggestions.append("• 您的心率偏低，建议适量有氧运动")
            elif hr > 100:
                suggestions.append("• 您的心率偏快，建议避免咖啡因和压力")
            
            # 基于HRV的建议
            if results.get("hrv_analysis", {}).get("rmssd", 0) < 30:
                suggestions.append("• 心率变异性较低，推荐每天进行深呼吸练习")
            
            # 基于ST段的建议
            st_status = results["wave_features"].get("st_segment", {}).get("status", "")
            if st_status in ["抬高", "压低"]:
                suggestions.append("• 检测到ST段改变，建议尽快就医检查")
            
            # 基于心律失常的建议
            if results.get("arrhythmia", {}).get("types", []):
                suggestions.append("• 检测到心律异常，建议咨询心脏专科医生")
            
            # 基于疾病风险的建议
            high_risk_diseases = [d for d, data in results.get("disease_risks", {}).items() 
                                if data["risk_level"] in ["中高", "高", "极高"]]
            if high_risk_diseases:
                suggestions.append(f"• 检测到{len(high_risk_diseases)}种中高风险疾病，建议尽快就医检查")
            
            return "\n".join(suggestions) if suggestions else "• 您的心脏指标在正常范围内，保持当前健康生活习惯即可"
        except:
            return "• 无法生成健康建议"


if __name__ == "__main__":
    # 测试代码
    processor = ECGProcessor()
    success, results, report = processor.analyze_ecg_file("test.dat")
    print(results)