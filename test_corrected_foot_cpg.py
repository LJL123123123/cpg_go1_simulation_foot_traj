#!/usr/bin/env python3
"""
测试修正后的足端轨迹CPG
Test the corrected foot trajectory CPG
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('/home/cpg_go1_simulation/src')

from cpg_go1_simulation.stein.foot_trajectory_cpg import FootTrajectoryCPG

def test_single_foot_trajectory():
    """测试单个足端的轨迹生成"""
    
    print("测试单个足端轨迹生成...")
    
    # 创建CPG实例
    cpg = FootTrajectoryCPG(
        before_ftype=1,  # walk
        after_ftype=1,
        total_time=2.0,
        toc=1.0,
        step_height=0.08,
        step_length=0.15,
        body_height=0.25,
        foot_spacing=0.3
    )
    
    # 生成时间序列
    time_steps = 1000
    total_time = 2.0
    time = np.linspace(0, total_time, time_steps)
    
    # 为每个足端生成轨迹
    trajectories = {}
    for foot_name in cpg.foot_names:
        traj = []
        for t in time:
            pos = cpg.generate_foot_position(foot_name, t)
            traj.append(pos)
        trajectories[foot_name] = np.array(traj)
    
    # 可视化轨迹
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Individual Foot Trajectory Analysis (Walk Gait)', fontsize=16)
    
    colors = ['red', 'blue', 'green', 'orange'] 
    foot_labels = ['Left Front', 'Right Front', 'Left Hind', 'Right Hind']
    
    for i, (foot_name, label) in enumerate(zip(cpg.foot_names, foot_labels)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        traj = trajectories[foot_name]
        
        # X-Z轨迹 (侧视图)
        ax.plot(traj[:, 0], traj[:, 2], color=colors[i], linewidth=2, label='Trajectory')
        ax.scatter(traj[0, 0], traj[0, 2], color='green', s=100, label='Start', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 2], color='red', s=100, label='End', zorder=5)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.set_title(f'{label} Foot Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('individual_foot_test.png', dpi=300, bbox_inches='tight')
    print("单足轨迹测试图已保存: individual_foot_test.png")
    plt.close()
    
    # 分析轨迹特征
    print("\n轨迹特征分析:")
    for foot_name in cpg.foot_names:
        traj = trajectories[foot_name]
        x_range = traj[:, 0].max() - traj[:, 0].min()
        z_range = traj[:, 2].max() - traj[:, 2].min()
        print(f"{foot_name}: X范围={x_range:.4f}m, Z范围={z_range:.4f}m")

def test_gait_coordination():
    """测试步态协调性"""
    
    print("\n测试步态协调性...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Gait Coordination Analysis', fontsize=16)
    
    gaits = ['walk', 'trot', 'pace', 'bound', 'pronk']
    gait_ids = [1, 2, 3, 4, 5]
    
    for gait_idx, (gait_name, gait_id) in enumerate(zip(gaits, gait_ids)):
        if gait_idx >= 5:  # 只显示5个步态
            break
            
        # 创建CPG实例
        cpg = FootTrajectoryCPG(
            before_ftype=gait_id,
            after_ftype=gait_id,
            total_time=2.0,
            toc=1.0,
            step_height=0.08,
            step_length=0.15,
            body_height=0.25,
            foot_spacing=0.3
        )
        
        # 生成轨迹数据
        time_steps = 1000
        time = np.linspace(0, 2.0, time_steps)
        
        # 选择合适的子图位置
        if gait_idx < 3:
            row, col = 0, gait_idx
        else:
            row, col = 1, gait_idx - 3
            
        ax = axes[row, col]
        
        colors = ['red', 'blue', 'green', 'orange']
        for foot_idx, foot_name in enumerate(cpg.foot_names):
            z_positions = []
            for t in time:
                pos = cpg.generate_foot_position(foot_name, t)
                z_positions.append(pos[2])
            
            ax.plot(time, z_positions, color=colors[foot_idx], 
                   label=foot_name, linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z Position (m)')
        ax.set_title(f'{gait_name.upper()} Gait')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏空的子图
    if len(gaits) < 6:
        axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('gait_coordination_test.png', dpi=300, bbox_inches='tight')
    print("步态协调测试图已保存: gait_coordination_test.png")
    plt.close()

def test_realistic_motion():
    """测试是否产生现实的运动模式"""
    
    print("\n测试现实运动模式...")
    
    # 创建walk步态CPG
    cpg = FootTrajectoryCPG(
        before_ftype=1,  # walk
        after_ftype=1,
        total_time=3.0,
        toc=1.0,
        step_height=0.06,  # 减小抬腿高度
        step_length=0.10,  # 减小步长
        body_height=0.25,
        foot_spacing=0.25
    )
    
    # 生成详细的轨迹数据
    time_steps = 1500
    time = np.linspace(0, 3.0, time_steps)
    
    # 记录所有足端轨迹
    all_trajectories = {}
    for foot_name in cpg.foot_names:
        traj = []
        for t in time:
            pos = cpg.generate_foot_position(foot_name, t)
            traj.append(pos)
        all_trajectories[foot_name] = np.array(traj)
    
    # 创建综合分析图
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D轨迹视图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    colors = ['red', 'blue', 'green', 'orange']
    for i, foot_name in enumerate(cpg.foot_names):
        traj = all_trajectories[foot_name]
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=colors[i], label=foot_name, linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Foot Trajectories')
    ax1.legend()
    
    # 2. 侧视图 (X-Z)
    ax2 = fig.add_subplot(2, 3, 2)
    for i, foot_name in enumerate(cpg.foot_names):
        traj = all_trajectories[foot_name]
        ax2.plot(traj[:, 0], traj[:, 2], color=colors[i], label=foot_name, linewidth=2)
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Z Position (m)') 
    ax2.set_title('Side View (X-Z)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    # 3. 俯视图 (X-Y)
    ax3 = fig.add_subplot(2, 3, 3)
    for i, foot_name in enumerate(cpg.foot_names):
        traj = all_trajectories[foot_name]
        ax3.plot(traj[:, 0], traj[:, 1], color=colors[i], label=foot_name, linewidth=2)
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Y Position (m)')
    ax3.set_title('Top View (X-Y)')
    ax3.legend()
    ax3.grid(True)
    ax3.set_aspect('equal')
    
    # 4. 足端高度时间序列
    ax4 = fig.add_subplot(2, 3, 4)
    for i, foot_name in enumerate(cpg.foot_names):
        traj = all_trajectories[foot_name]
        ax4.plot(time, traj[:, 2], color=colors[i], label=foot_name, linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Z Position (m)')
    ax4.set_title('Foot Height vs Time')
    ax4.legend()
    ax4.grid(True)
    
    # 5. X位置时间序列
    ax5 = fig.add_subplot(2, 3, 5)
    for i, foot_name in enumerate(cpg.foot_names):
        traj = all_trajectories[foot_name]
        ax5.plot(time, traj[:, 0], color=colors[i], label=foot_name, linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('X Position (m)')
    ax5.set_title('Forward Motion vs Time')
    ax5.legend()
    ax5.grid(True)
    
    # 6. 步态相位分析
    ax6 = fig.add_subplot(2, 3, 6)
    lf_traj = all_trajectories['LF']
    rf_traj = all_trajectories['RF']
    ax6.plot(lf_traj[:, 2], rf_traj[:, 2], color='purple', linewidth=2)
    ax6.set_xlabel('Left Front Z (m)')
    ax6.set_ylabel('Right Front Z (m)')
    ax6.set_title('Phase Plot (LF vs RF)')
    ax6.grid(True)
    ax6.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('realistic_motion_test.png', dpi=300, bbox_inches='tight')
    print("现实运动测试图已保存: realistic_motion_test.png")
    plt.close()
    
    # 分析运动质量
    print("\n运动质量分析:")
    
    # 检查步长
    lf_traj = all_trajectories['LF']
    x_range = lf_traj[:, 0].max() - lf_traj[:, 0].min()
    z_range = lf_traj[:, 2].max() - lf_traj[:, 2].min()
    
    print(f"步长: {x_range:.4f} m")
    print(f"抬腿高度: {z_range:.4f} m")
    
    # 检查周期性
    z_data = lf_traj[:, 2]
    peaks = []
    for i in range(1, len(z_data)-1):
        if z_data[i] > z_data[i-1] and z_data[i] > z_data[i+1]:
            if z_data[i] > z_data.min() + 0.5 * z_range:
                peaks.append(i)
    
    if len(peaks) > 1:
        avg_period = np.mean(np.diff(peaks)) / (time_steps / 3.0)  # 转换为秒
        frequency = 1.0 / avg_period if avg_period > 0 else 0
        print(f"步频: {frequency:.2f} Hz")
    else:
        print("步频: 无法检测到周期性")
    
    # 检查相位关系
    lf_z = all_trajectories['LF'][:, 2]
    rf_z = all_trajectories['RF'][:, 2]
    correlation = np.corrcoef(lf_z, rf_z)[0, 1]
    print(f"左前与右前足相关性: {correlation:.3f}")
    
    return all_trajectories

if __name__ == "__main__":
    print("开始测试修正后的足端轨迹CPG...")
    
    # 1. 测试单足轨迹
    test_single_foot_trajectory()
    
    # 2. 测试步态协调
    test_gait_coordination()
    
    # 3. 测试现实运动
    trajectories = test_realistic_motion()
    
    print(f"\n测试完成! 生成的图像文件:")
    print("- individual_foot_test.png: 单足轨迹测试")
    print("- gait_coordination_test.png: 步态协调测试")
    print("- realistic_motion_test.png: 现实运动测试")
    print("\n如果这些轨迹看起来合理，接下来可以集成到完整的CPG系统中。")