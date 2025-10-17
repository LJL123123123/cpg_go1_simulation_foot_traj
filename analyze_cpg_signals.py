#!/usr/bin/env python3
"""
分析CPG神经网络生成的信号特征，并展示如何转换为步态数据
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_cpg_signals():
    """分析CPG信号的特征"""
    
    # 读取生成的CPG数据
    data_file = Path('data/cpg_data/gait_data/cpg_walk_2.0s.csv')
    if not data_file.exists():
        print("CPG数据文件不存在，请先生成数据")
        return
    
    df = pd.read_csv(data_file, header=None)
    print("=== CPG神经网络信号分析 ===")
    print(f"数据形状: {df.shape}")
    print(f"采样频率: 500Hz (每0.002秒一个样本)")
    print(f"总时长: {df.shape[0]/500:.1f}秒")
    
    # 分离神经元输出和导数
    neuron_outputs = df.iloc[:, :8].values  # x1-x8: 8个神经元的输出
    neuron_derivatives = df.iloc[:, 8:16].values  # dx1-dx8: 8个神经元的导数
    
    # 创建时间轴
    time = np.arange(len(df)) / 500.0  # 转换为秒
    
    print("\n=== 神经元输出特征 ===")
    print(f"神经元输出范围: [{neuron_outputs.min():.4f}, {neuron_outputs.max():.4f}]")
    print("神经元输出统计:")
    for i in range(8):
        print(f"  神经元{i+1}: 均值={neuron_outputs[:,i].mean():.4f}, "
              f"标准差={neuron_outputs[:,i].std():.4f}")
    
    print("\n=== 神经元导数特征 ===")
    print(f"神经元导数范围: [{neuron_derivatives.min():.4f}, {neuron_derivatives.max():.4f}]")
    
    # 分析神经元的振荡模式
    print("\n=== 振荡模式分析 ===")
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. 绘制神经元输出信号
    axes[0].set_title('8-神经元CPG网络输出 (Walk步态)', fontsize=14)
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for i in range(8):
        axes[0].plot(time, neuron_outputs[:, i], 
                    label=f'神经元{i+1}', color=colors[i], linewidth=1.5)
    
    axes[0].set_xlabel('时间 (秒)')
    axes[0].set_ylabel('神经元输出')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 绘制髋关节和膝关节神经元的对比
    axes[1].set_title('髋关节vs膝关节神经元 (前4个=髋关节, 后4个=膝关节)', fontsize=14)
    
    # 髋关节神经元 (1-4)
    hip_mean = neuron_outputs[:, :4].mean(axis=1)
    axes[1].plot(time, hip_mean, label='髋关节神经元平均', color='red', linewidth=2)
    
    # 膝关节神经元 (5-8)  
    knee_mean = neuron_outputs[:, 4:8].mean(axis=1)
    axes[1].plot(time, knee_mean, label='膝关节神经元平均', color='blue', linewidth=2)
    
    axes[1].set_xlabel('时间 (秒)')
    axes[1].set_ylabel('神经元输出')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 绘制相位关系 (神经元1 vs 神经元3，显示左右腿的相位差)
    axes[2].set_title('步态相位关系 (神经元1 vs 神经元3 - 显示左右腿协调)', fontsize=14)
    axes[2].plot(time, neuron_outputs[:, 0], label='神经元1 (左前腿髋关节)', color='green', linewidth=2)
    axes[2].plot(time, neuron_outputs[:, 2], label='神经元3 (右前腿髋关节)', color='orange', linewidth=2)
    axes[2].set_xlabel('时间 (秒)')
    axes[2].set_ylabel('神经元输出')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cpg_signal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return neuron_outputs, neuron_derivatives, time

def demonstrate_gait_conversion():
    """演示如何将CPG信号转换为步态数据"""
    
    print("\n" + "="*50)
    print("=== CPG信号到步态数据的转换过程 ===")
    print("="*50)
    
    # 1. 读取CPG数据
    data_file = Path('data/cpg_data/gait_data/cpg_walk_2.0s.csv')
    df = pd.read_csv(data_file, header=None)
    cpg_data = df.values
    
    print("步骤1: 原始CPG神经信号")
    print(f"  - 8个神经元输出 (x1-x8): 范围[0,1]的连续值")
    print(f"  - 8个神经元导数 (dx1-dx8): 变化率信息")
    print(f"  - 采样频率: 500Hz")
    print(f"  - 数据示例 (第1个时间步):")
    print(f"    神经元输出: {cpg_data[0, :8].round(4)}")
    print(f"    神经元导数: {cpg_data[0, 8:16].round(4)}")
    
    print("\n步骤2: 数据预处理")
    print("  - 提取神经元输出值和导数")
    print("  - 添加步态类型编码 (Walk=1)")
    print("  - 添加关节编码 (8个关节对应8个神经元)")
    print("  - 添加步态比例参数 (stance_length_ratio)")
    
    # 模拟数据预处理过程
    sample_cpg_state = cpg_data[100, :]  # 取第100个时间步作为示例
    
    print(f"\n步骤3: 神经网络输入特征构造 (示例)")
    
    # 模拟特征构造过程（基于process_network.py的逻辑）
    gait_onehot = [1, 0, 0, 0, 0]  # walk步态的one-hot编码
    
    for i in range(8):
        # 关节编码 (3-bit)
        joint_onehot = [
            [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0],
            [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]
        ][i]
        
        # 构造特征向量
        features = (joint_onehot +           # 关节编码 (3-bit)
                   gait_onehot +            # 步态编码 (5-bit) 
                   [0.8] +                  # 步态比例参数
                   [sample_cpg_state[i]] +  # 神经元输出值
                   [sample_cpg_state[i+8] / 50])  # 神经元导数/50
        
        if i == 0:  # 只显示第一个关节的示例
            print(f"  关节{i+1}特征向量 ({len(features)}维):")
            print(f"    关节编码: {joint_onehot}")
            print(f"    步态编码: {gait_onehot}")  
            print(f"    步态比例: 0.8")
            print(f"    神经元值: {sample_cpg_state[i]:.4f}")
            print(f"    神经元导数: {sample_cpg_state[i+8]/50:.4f}")
    
    print(f"\n步骤4: 神经网络预测")
    print("  - 输入: 8组特征向量 (每组12维)")
    print("  - 网络: 多层感知器 (MLP)")
    print("  - 输出: 8个关节角度预测值")
    
    print(f"\n步骤5: 关节角度映射")
    print("  - 神经元1-4 → 髋关节角度 (4条腿)")
    print("  - 神经元5-8 → 膝关节角度 (4条腿)")  
    print("  - 映射关系:")
    print("    神经元1 → 左后腿髋关节")
    print("    神经元2 → 右后腿髋关节") 
    print("    神经元3 → 右前腿髋关节")
    print("    神经元4 → 左前腿髋关节")
    print("    神经元5 → 左后腿膝关节")
    print("    神经元6 → 右后腿膝关节")
    print("    神经元7 → 右前腿膝关节") 
    print("    神经元8 → 左前腿膝关节")
    
    print(f"\n步骤6: 机器人控制")
    print("  - 将预测的关节角度发送给机器人")
    print("  - 通过位置控制实现步态运动")
    print("  - 500Hz的控制频率保证平滑运动")

def analyze_gait_patterns():
    """分析不同步态的CPG模式"""
    
    print("\n" + "="*50)
    print("=== 不同步态的CPG神经模式特征 ===")
    print("="*50)
    
    # 从代码中提取的步态参数
    gait_params = {
        "walk": {"a_hip": 10, "f_hip": 40, "k1_hip": 0, "k2_hip": 0},
        "trot": {"a_hip": 11, "f_hip": 41, "k1_hip": 0.085, "k2_hip": 56},
        "pace": {"a_hip": 11, "f_hip": 41, "k1_hip": 0.04, "k2_hip": 54},
        "bound": {"a_hip": 16, "f_hip": 50, "k1_hip": 0.1, "k2_hip": 59}, 
        "pronk": {"a_hip": 22, "f_hip": 65, "k1_hip": 0.3, "k2_hip": 60}
    }
    
    print("各步态的神经网络参数差异:")
    print("参数说明:")
    print("  - a_hip: 髋关节神经元的激活强度")
    print("  - f_hip: 髋关节神经元的驱动频率") 
    print("  - k1_hip: 髋关节神经元的耦合强度")
    print("  - k2_hip: 髋关节神经元的振荡频率")
    print()
    
    for gait, params in gait_params.items():
        print(f"{gait.upper()}步态:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        print()
    
    print("步态特征分析:")
    print("1. WALK (行走): 最基础的步态，参数较小，节奏较慢")
    print("2. TROT (小跑): 中等强度，开始有耦合振荡 (k1_hip=0.085)")
    print("3. PACE (溜蹄): 类似trot但耦合参数不同 (k1_hip=0.04)")
    print("4. BOUND (跳跃): 更强的激活 (a_hip=16, f_hip=50)")
    print("5. PRONK (四腿同时跳): 最强激活 (a_hip=22, f_hip=65)")

if __name__ == "__main__":
    print("分析CPG神经网络信号...")
    
    # 分析信号特征
    neuron_outputs, neuron_derivatives, time = analyze_cpg_signals()
    
    # 演示转换过程
    demonstrate_gait_conversion()
    
    # 分析步态模式
    analyze_gait_patterns()
    
    print(f"\n总结:")
    print("CPG神经网络通过8个相互耦合的Stein振荡器生成周期性信号")
    print("这些信号经过神经网络处理后转换为机器人的关节角度")
    print("不同的网络参数产生不同的步态模式")
    print("信号分析图已保存为: cpg_signal_analysis.png")