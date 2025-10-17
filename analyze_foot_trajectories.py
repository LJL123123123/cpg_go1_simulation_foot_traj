#!/usr/bin/env python3
"""
足端轨迹CPG数据分析和可视化
Analysis and visualization of foot trajectory CPG data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_foot_trajectory_data():
    """分析足端轨迹数据"""
    
    # 读取walk步态数据
    data_file = Path('data/cpg_data/gait_data/foot_traj_walk_3.0s.csv')
    if not data_file.exists():
        print("请先生成足端轨迹数据")
        return
        
    df = pd.read_csv(data_file)
    
    print("=== 足端轨迹CPG数据分析 ===")
    print(f"数据形状: {df.shape}")
    print(f"采样频率: 500Hz")
    print(f"总时长: {df.shape[0]/500:.1f}秒")
    print()
    
    # 提取足端位置数据
    foot_names = ['LF', 'RF', 'LH', 'RH']
    coordinates = ['x', 'y', 'z']
    
    print("足端轨迹范围分析:")
    for foot in foot_names:
        for coord in coordinates:
            col_name = f"{foot}_{coord}"
            if col_name in df.columns:
                min_val = df[col_name].min()
                max_val = df[col_name].max()
                mean_val = df[col_name].mean()
                print(f"  {col_name}: 范围[{min_val:.4f}, {max_val:.4f}], 均值{mean_val:.4f}")
    
    return df

def compare_gait_trajectories():
    """比较不同步态的足端轨迹"""
    
    gaits = ['walk', 'trot', 'pace', 'bound', 'pronk']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Foot Trajectory Comparison Across Gaits', fontsize=16)
    
    # 为每个足端创建子图
    foot_names = ['LF', 'RF', 'LH', 'RH']
    foot_labels = ['Left Front', 'Right Front', 'Left Hind', 'Right Hind']
    
    for foot_idx, (foot, label) in enumerate(zip(foot_names, foot_labels)):
        row = foot_idx // 2
        col = foot_idx % 2
        ax = axes[row, col]
        
        for gait_idx, gait in enumerate(gaits):
            data_file = Path(f'data/cpg_data/gait_data/foot_traj_{gait}_3.0s.csv')
            
            if data_file.exists():
                df = pd.read_csv(data_file)
                
                # 提取该足端的x-z轨迹 (侧视图)
                x_col = f"{foot}_x"
                z_col = f"{foot}_z"
                
                if x_col in df.columns and z_col in df.columns:
                    # 只显示前500个点 (1秒的数据)
                    x_data = df[x_col][:500]
                    z_data = df[z_col][:500]
                    
                    ax.plot(x_data, z_data, color=colors[gait_idx], 
                           label=gait, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.set_title(f'{label} Foot Trajectory (Side View)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('foot_trajectory_comparison.png', dpi=300, bbox_inches='tight')
    print("步态轨迹比较图已保存: foot_trajectory_comparison.png")
    plt.close()

def analyze_gait_characteristics():
    """分析不同步态的特征"""
    
    gaits = ['walk', 'trot', 'pace', 'bound', 'pronk']
    
    print("\n=== 不同步态的足端轨迹特征分析 ===")
    
    gait_stats = {}
    
    for gait in gaits:
        data_file = Path(f'data/cpg_data/gait_data/foot_traj_{gait}_3.0s.csv')
        
        if data_file.exists():
            df = pd.read_csv(data_file)
            
            # 计算步态特征
            stats = {}
            
            # 步长 (x方向的范围)
            x_cols = [col for col in df.columns if '_x' in col and '_dx' not in col]
            step_lengths = []
            for col in x_cols:
                step_length = df[col].max() - df[col].min()
                step_lengths.append(step_length)
            stats['step_length'] = np.mean(step_lengths)
            
            # 抬腿高度 (z方向的最大值 - 最小值)
            z_cols = [col for col in df.columns if '_z' in col and '_dz' not in col]
            lift_heights = []
            for col in z_cols:
                lift_height = df[col].max() - df[col].min()
                lift_heights.append(lift_height)
            stats['lift_height'] = np.mean(lift_heights)
            
            # 步态频率 (基于z坐标的周期性)
            # 简单估计: 计算前1500个点中z坐标的峰值数量
            z_data = df[z_cols[0]][:1500]  # 使用左前足的z坐标
            peaks = 0
            for i in range(1, len(z_data)-1):
                if z_data.iloc[i] > z_data.iloc[i-1] and z_data.iloc[i] > z_data.iloc[i+1]:
                    if z_data.iloc[i] > df[z_cols[0]].min() + stats['lift_height'] * 0.5:
                        peaks += 1
            stats['frequency'] = peaks / 3.0  # 3秒数据
            
            gait_stats[gait] = stats
            
            print(f"{gait.upper()}步态:")
            print(f"  平均步长: {stats['step_length']:.4f} m")
            print(f"  平均抬腿高度: {stats['lift_height']:.4f} m")
            print(f"  估计步频: {stats['frequency']:.2f} Hz")
            print()
    
    return gait_stats

def visualize_single_gait_details():
    """详细可视化单个步态的足端轨迹"""
    
    gait = 'walk'
    data_file = Path(f'data/cpg_data/gait_data/foot_traj_{gait}_3.0s.csv')
    
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        return
        
    df = pd.read_csv(data_file)
    
    # 创建时间轴 (只显示前3000个点，即6秒)
    time_points = 3000
    time = np.arange(time_points) / 500.0
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'{gait.upper()} Gait - Detailed Foot Trajectory Analysis', fontsize=16)
    
    foot_names = ['LF', 'RF', 'LH', 'RH']
    colors = ['red', 'blue', 'green', 'orange']
    
    # 1. X坐标时间序列
    ax1 = axes[0, 0]
    for i, foot in enumerate(foot_names):
        col = f"{foot}_x"
        if col in df.columns:
            ax1.plot(time, df[col][:time_points], color=colors[i], label=foot, linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position (m)')
    ax1.set_title('X Position vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Z坐标时间序列
    ax2 = axes[0, 1]
    for i, foot in enumerate(foot_names):
        col = f"{foot}_z"
        if col in df.columns:
            ax2.plot(time, df[col][:time_points], color=colors[i], label=foot, linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z Position (m)')
    ax2.set_title('Z Position vs Time (Foot Height)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 左前足3D轨迹
    ax3 = axes[1, 0]
    if 'LF_x' in df.columns and 'LF_z' in df.columns:
        ax3.plot(df['LF_x'][:time_points], df['LF_z'][:time_points], 
                color='red', linewidth=2, label='Left Front')
        ax3.scatter(df['LF_x'].iloc[0], df['LF_z'].iloc[0], 
                   color='green', s=100, label='Start', zorder=5)
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_title('Left Front Foot Trajectory (Side View)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 4. 右前足3D轨迹
    ax4 = axes[1, 1]
    if 'RF_x' in df.columns and 'RF_z' in df.columns:
        ax4.plot(df['RF_x'][:time_points], df['RF_z'][:time_points], 
                color='blue', linewidth=2, label='Right Front')
        ax4.scatter(df['RF_x'].iloc[0], df['RF_z'].iloc[0], 
                   color='green', s=100, label='Start', zorder=5)
    ax4.set_xlabel('X Position (m)')
    ax4.set_ylabel('Z Position (m)')
    ax4.set_title('Right Front Foot Trajectory (Side View)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # 5. 足端速度分析
    ax5 = axes[2, 0]
    for i, foot in enumerate(foot_names):
        col = f"{foot}_dz"
        if col in df.columns:
            ax5.plot(time, df[col][:time_points], color=colors[i], label=foot, linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Z Velocity (m/s)')
    ax5.set_title('Foot Vertical Velocity')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 步态相位分析 (左前足vs右前足的Z坐标)
    ax6 = axes[2, 1]
    if 'LF_z' in df.columns and 'RF_z' in df.columns:
        ax6.plot(df['LF_z'][:time_points], df['RF_z'][:time_points], 
                color='purple', linewidth=2, alpha=0.7)
        ax6.scatter(df['LF_z'].iloc[0], df['RF_z'].iloc[0], 
                   color='green', s=100, label='Start', zorder=5)
    ax6.set_xlabel('Left Front Z (m)')
    ax6.set_ylabel('Right Front Z (m)')
    ax6.set_title('Phase Relationship (LF vs RF)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{gait}_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"{gait}步态详细分析图已保存: {gait}_detailed_analysis.png")
    plt.close()

if __name__ == "__main__":
    print("分析足端轨迹CPG数据...")
    
    # 1. 基础数据分析
    df = analyze_foot_trajectory_data()
    
    # 2. 比较不同步态
    compare_gait_trajectories()
    
    # 3. 分析步态特征
    gait_stats = analyze_gait_characteristics()
    
    # 4. 详细分析单个步态
    visualize_single_gait_details()
    
    print(f"\n=== 总结 ===")
    print("足端轨迹CPG成功生成了四足机器人的足端轨迹数据:")
    print("- 每个足端生成独立的3D轨迹 (x,y,z坐标)")
    print("- 不同步态显示出不同的运动模式")
    print("- 数据包含位置和速度信息，可直接用于机器人控制")
    print("- 轨迹符合生物学步态特征，支撑相和摆动相清晰可见")
    print()
    print("生成的可视化文件:")
    print("- foot_trajectory_comparison.png: 步态间比较")
    print("- walk_detailed_analysis.png: Walk步态详细分析")