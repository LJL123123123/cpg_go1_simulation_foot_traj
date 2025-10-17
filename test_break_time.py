#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试带有break时间的足端轨迹CPG
验证足端落地后的稳定时间是否按预期工作
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cpg_go1_simulation.stein.foot_trajectory_cpg import FootTrajectoryCPG

def test_break_time_effect():
    """测试break时间对步态稳定性的影响"""
    
    print("=== 测试足端break时间功能 ===\n")
    
    # 测试参数
    break_times = [0.0, 0.05, 0.1]  # 不同的break时间
    gait_types = [(1, "Walk"), (2, "Trot")]
    
    fig, axes = plt.subplots(len(break_times), 2, figsize=(15, 12))
    fig.suptitle('足端Break时间对步态稳定性的影响', fontsize=16)
    
    for break_idx, break_time in enumerate(break_times):
        print(f"测试break时间: {break_time}s")
        
        for gait_idx, (gait_id, gait_name) in enumerate(gait_types):
            print(f"  步态: {gait_name}")
            
            # 创建CPG实例
            cpg = FootTrajectoryCPG(
                before_ftype=gait_id,
                after_ftype=gait_id,
                total_time=3.0,
                toc=4.0,
                break_time=break_time
            )
            
            # 生成轨迹数据
            duration = 3.0
            dt = 0.01
            times = np.arange(0, duration, dt)
            
            # 记录足端高度和速度
            foot_heights = {name: [] for name in cpg.foot_names}
            foot_velocities = {name: [] for name in cpg.foot_names}
            stability_periods = []  # 记录稳定期
            
            for t in times:
                current_stability = 0
                for foot_name in cpg.foot_names:
                    pos = cpg.generate_foot_position(foot_name, t)
                    
                    # 计算前一时刻的位置来估算速度
                    if t > dt:
                        prev_pos = cpg.generate_foot_position(foot_name, t - dt)
                        velocity = np.linalg.norm(pos - prev_pos) / dt
                    else:
                        velocity = 0.0
                    
                    foot_heights[foot_name].append(pos[2])
                    foot_velocities[foot_name].append(velocity)
                    
                    # 检查是否在break期（低速且接触地面）
                    if pos[2] <= cpg.foot_base_positions[foot_name][2] + 0.01 and velocity < 0.1:
                        current_stability += 1
                
                stability_periods.append(current_stability)
            
            # 绘制足端高度
            ax = axes[break_idx, gait_idx]
            colors = ['red', 'blue', 'green', 'orange']
            
            for i, foot_name in enumerate(cpg.foot_names):
                ax.plot(times, foot_heights[foot_name], color=colors[i], 
                       label=foot_name, linewidth=1.5)
            
            # 标记稳定期
            ax2 = ax.twinx()
            ax2.plot(times, stability_periods, 'k--', alpha=0.7, 
                    label=f'稳定足数 (break={break_time}s)')
            ax2.set_ylabel('稳定足数', color='k')
            
            ax.set_title(f'{gait_name} - Break时间: {break_time}s')
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('足端高度 (m)')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # 计算稳定性指标
            avg_stability = np.mean(stability_periods)
            stability_variance = np.var(stability_periods)
            
            print(f"    平均稳定足数: {avg_stability:.2f}")
            print(f"    稳定性方差: {stability_variance:.4f}")
            
    plt.tight_layout()
    plt.savefig('/home/cpg_go1_simulation/break_time_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def analyze_break_time_details():
    """详细分析break时间的实现效果"""
    
    print("\n=== 详细分析break时间效果 ===\n")
    
    # 使用Walk步态进行详细分析
    cpg = FootTrajectoryCPG(
        before_ftype=1,  # Walk
        after_ftype=1,
        total_time=2.0,
        toc=3.0,
        break_time=0.05
    )
    
    duration = 2.0
    dt = 0.005  # 更高精度
    times = np.arange(0, duration, dt)
    
    # 分析单个足端的详细行为
    foot_name = "LF"  # 分析左前足
    
    positions = []
    velocities = []
    phase_info = []
    is_break_periods = []
    
    for t in times:
        pos = cpg.generate_foot_position(foot_name, t)
        positions.append(pos)
        
        # 计算速度
        if len(positions) > 1:
            velocity = np.linalg.norm(positions[-1] - positions[-2]) / dt
        else:
            velocity = 0.0
        velocities.append(velocity)
        
        # 获取相位信息
        foot_phases = cpg.get_all_foot_phases(t)
        foot_phase = foot_phases[foot_name]
        phase_info.append(foot_phase)
        
        # 判断是否在break期
        stance_progress = foot_phase['phase_ratio'] / cpg.duty_factor if foot_phase['is_stance'] else 1.0
        cycle_time = 2 * np.pi / cpg.omega
        stance_time = cpg.duty_factor * cycle_time
        break_ratio_in_stance = cpg.break_time / stance_time
        
        is_break = foot_phase['is_stance'] and stance_progress <= break_ratio_in_stance
        is_break_periods.append(is_break)
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # 可视化详细分析
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{foot_name}足端Break时间详细分析', fontsize=16)
    
    # 1. 足端轨迹 (X-Z平面)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(positions[:, 0], positions[:, 2], 
                         c=is_break_periods, cmap='RdYlBu', s=20)
    ax1.set_xlabel('X坐标 (m)')
    ax1.set_ylabel('Z坐标 (m)')
    ax1.set_title('足端轨迹 (红=Break期, 蓝=其他)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1)
    
    # 2. 足端高度时间序列
    ax2 = axes[0, 1]
    ax2.plot(times, positions[:, 2], 'b-', linewidth=2, label='足端高度')
    
    # 标记break期
    break_mask = np.array(is_break_periods)
    if np.any(break_mask):
        ax2.fill_between(times, positions[:, 2], alpha=0.3, 
                        where=break_mask, color='red', label='Break期')
    
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('Z坐标 (m)')
    ax2.set_title('足端高度变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 速度时间序列
    ax3 = axes[1, 0]
    ax3.plot(times, velocities, 'g-', linewidth=2, label='足端速度')
    
    # 标记break期
    if np.any(break_mask):
        ax3.fill_between(times, 0, max(velocities), alpha=0.3,
                        where=break_mask, color='red', label='Break期')
    
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('速度 (m/s)')
    ax3.set_title('足端移动速度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 相位和状态分析
    ax4 = axes[1, 1]
    phase_ratios = [info['phase_ratio'] for info in phase_info]
    is_stance = [info['is_stance'] for info in phase_info]
    
    ax4.plot(times, phase_ratios, 'purple', linewidth=2, label='相位比例')
    ax4.fill_between(times, 0, 1, where=is_stance, alpha=0.3, 
                    color='blue', label='支撑相')
    if np.any(break_mask):
        ax4.fill_between(times, 0, 1, where=break_mask, alpha=0.5,
                        color='red', label='Break期')
    
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('相位比例')
    ax4.set_title('步态相位分析')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/cpg_go1_simulation/break_time_detailed_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 统计break时间的效果
    break_periods = np.sum(break_mask) * dt
    total_stance_time = np.sum([info['is_stance'] for info in phase_info]) * dt
    
    print(f"总break时间: {break_periods:.3f}s")
    print(f"总支撑时间: {total_stance_time:.3f}s") 
    print(f"Break时间占支撑时间比例: {break_periods/total_stance_time*100:.1f}%")
    print(f"Break期平均速度: {np.mean(np.array(velocities)[break_mask]):.4f} m/s")
    print(f"非Break期平均速度: {np.mean(np.array(velocities)[~break_mask]):.4f} m/s")

def generate_stable_gait_data():
    """生成带稳定break时间的步态数据"""
    
    print("\n=== 生成稳定步态数据 ===\n")
    
    gait_types = {
        1: "walk",
        2: "trot"
    }
    
    for gait_id, gait_name in gait_types.items():
        print(f"生成 {gait_name} 稳定步态数据...")
        
        # 创建带break时间的CPG
        cpg = FootTrajectoryCPG(
            before_ftype=gait_id,
            after_ftype=gait_id,
            total_time=5.0,
            toc=6.0,
            break_time=0.05  # 50ms稳定时间
        )
        
        # 导出数据
        cpg.export_csv()
        
        # 生成可视化
        save_path = f'/home/cpg_go1_simulation/stable_{gait_name}_with_break.png'
        cpg.plot_foot_trajectories(save_path)
        
        print(f"  ✅ {gait_name} 数据已生成")

if __name__ == "__main__":
    
    # 1. 测试break时间对步态的影响
    test_break_time_effect()
    
    # 2. 详细分析break时间的实现
    analyze_break_time_details()
    
    # 3. 生成稳定的步态数据
    generate_stable_gait_data()
    
    print("\n=== Break时间功能测试完成 ===")
    print("生成的文件:")
    print("- break_time_analysis.png")
    print("- break_time_detailed_analysis.png")
    print("- stable_walk_with_break.png")
    print("- stable_trot_with_break.png")
    print("- 带break时间的CSV数据文件")