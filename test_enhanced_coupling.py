#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试增强耦合约束的足端轨迹CPG生成器
验证Walk和Trot步态的生物学约束是否正确执行
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cpg_go1_simulation.stein.foot_trajectory_cpg import FootTrajectoryCPG

def test_gait_constraints():
    """测试步态约束条件"""
    
    # 初始化CPG
    cpg = FootTrajectoryCPG(
        before_ftype=1,  # 初始步态
        after_ftype=1,   # 目标步态
        total_time=5.0,  # 转换时间
        toc=2.0         # 转换开始时间
    )
    
    # 测试参数
    duration = 4.0  # 测试4秒
    dt = 0.02      # 50Hz采样率
    time_steps = int(duration / dt)
    times = np.linspace(0, duration, time_steps)
    
    # 测试两种关键步态
    test_gaits = {
        1: "Walk - 同时只能有一个足端腾空",
        2: "Trot - 只能有对角足端同时腾空"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('增强耦合约束的步态验证', fontsize=16)
    
    for gait_idx, (ftype, gait_name) in enumerate(test_gaits.items()):
        print(f"\n测试 {gait_name}")
        
        # 设置步态
        cpg.before_ftype = ftype
        cpg.frequency, cpg.duty_factor, cpg.amplitude = cpg.get_foot_trajectory_params(ftype)
        cpg.omega = 2 * np.pi * cpg.frequency
        
        # 记录数据
        foot_states = {name: [] for name in cpg.foot_names}
        airborne_count = []
        constraint_violations = []
        
        for t in times:
            # 获取所有足端的相位信息
            foot_phases = cpg.get_all_foot_phases(t)
            foot_phases = cpg.enforce_gait_constraints(foot_phases, "LF")  # 应用约束
            
            # 记录每个足端的状态
            current_states = {}
            airborne_feet = []
            
            for foot_name in cpg.foot_names:
                is_stance = foot_phases[foot_name]['is_stance']
                current_states[foot_name] = 0 if is_stance else 1  # 0=支撑，1=腾空
                foot_states[foot_name].append(current_states[foot_name])
                
                if not is_stance:
                    airborne_feet.append(foot_name)
            
            # 统计腾空足端数量
            num_airborne = len(airborne_feet)
            airborne_count.append(num_airborne)
            
            # 检查约束违反
            violation = False
            if ftype == 1:  # Walk: 最多1个足端腾空
                violation = num_airborne > 1
            elif ftype == 2:  # Trot: 最多2个足端腾空，且必须是对角
                if num_airborne > 2:
                    violation = True
                elif num_airborne == 2:
                    # 检查是否是对角对
                    diagonal_pairs = [("LF", "RH"), ("RF", "LH")]
                    is_diagonal = any(
                        set(airborne_feet) == set(pair) for pair in diagonal_pairs
                    )
                    if not is_diagonal:
                        violation = True
            
            constraint_violations.append(violation)
        
        # 绘制结果
        ax1 = axes[gait_idx, 0]
        ax2 = axes[gait_idx, 1]
        
        # 足端状态时间序列
        for i, foot_name in enumerate(cpg.foot_names):
            ax1.plot(times, np.array(foot_states[foot_name]) + i * 1.2, 
                    label=foot_name, linewidth=2)
        
        ax1.set_title(f'{gait_name} - 足端状态')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('足端 (0=支撑, 1=腾空)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks(np.arange(len(cpg.foot_names)) * 1.2)
        ax1.set_yticklabels(cpg.foot_names)
        
        # 腾空足端数量
        ax2.plot(times, airborne_count, 'b-', linewidth=2, label='腾空足端数')
        
        # 标记约束违反
        violation_times = times[np.array(constraint_violations)]
        if len(violation_times) > 0:
            ax2.scatter(violation_times, 
                       np.array(airborne_count)[np.array(constraint_violations)],
                       color='red', s=50, label='约束违反', zorder=5)
        
        # 添加约束线
        if ftype == 1:  # Walk
            ax2.axhline(y=1, color='g', linestyle='--', alpha=0.7, label='最大允许(Walk=1)')
        elif ftype == 2:  # Trot
            ax2.axhline(y=2, color='g', linestyle='--', alpha=0.7, label='最大允许(Trot=2)')
        
        ax2.set_title(f'{gait_name} - 腾空足端数量')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('腾空足端数量')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.5, 4.5)
        
        # 统计结果
        total_violations = sum(constraint_violations)
        violation_rate = total_violations / len(times) * 100
        
        print(f"  约束违反次数: {total_violations}/{len(times)} ({violation_rate:.2f}%)")
        print(f"  平均腾空足端数: {np.mean(airborne_count):.2f}")
        print(f"  最大腾空足端数: {max(airborne_count)}")
    
    plt.tight_layout()
    plt.savefig('/home/cpg_go1_simulation/gait_constraint_validation.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return True

def analyze_phase_coupling():
    """分析相位耦合效果"""
    
    cpg = FootTrajectoryCPG(
        before_ftype=1,  # 初始步态
        after_ftype=2,   # 目标步态
        total_time=5.0,  # 转换时间
        toc=3.0         # 转换开始时间
    )
    
    # 生成Walk和Trot的相位分析
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
    for gait_idx, (ftype, gait_name) in enumerate([(1, "Walk"), (2, "Trot")]):
        cpg.before_ftype = ftype
        cpg.frequency, cpg.duty_factor, cpg.amplitude = cpg.get_foot_trajectory_params(ftype)
        cpg.omega = 2 * np.pi * cpg.frequency
        
        duration = 3.0
        dt = 0.01
        times = np.arange(0, duration, dt)
        
        # 记录相位数据
        phases = {name: [] for name in cpg.foot_names}
        stance_states = {name: [] for name in cpg.foot_names}
        
        for t in times:
            foot_phases = cpg.get_all_foot_phases(t)
            foot_phases = cpg.enforce_gait_constraints(foot_phases, "LF")
            
            for foot_name in cpg.foot_names:
                phases[foot_name].append(foot_phases[foot_name]['phase_ratio'])
                stance_states[foot_name].append(foot_phases[foot_name]['is_stance'])
        
        # 绘制相位图
        ax1 = axes[gait_idx, 0]
        for foot_name in cpg.foot_names:
            ax1.plot(times, phases[foot_name], label=foot_name, linewidth=2)
        
        ax1.set_title(f'{gait_name} - 相位同步')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('相位比例')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制步态图
        ax2 = axes[gait_idx, 1]
        for i, foot_name in enumerate(cpg.foot_names):
            stance_array = np.array(stance_states[foot_name])
            # 创建步态图：支撑相=1，腾空相=0
            ax2.fill_between(times, i, i+0.8, where=stance_array, 
                           alpha=0.7, label=f'{foot_name} 支撑相')
        
        ax2.set_title(f'{gait_name} - 步态图')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('足端')
        ax2.set_yticks(np.arange(len(cpg.foot_names)) + 0.4)
        ax2.set_yticklabels(cpg.foot_names)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/cpg_go1_simulation/phase_coupling_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("=== 足端轨迹CPG增强耦合约束测试 ===")
    
    print("\n1. 测试步态约束条件...")
    test_gait_constraints()
    
    print("\n2. 分析相位耦合效果...")
    analyze_phase_coupling()
    
    print("\n测试完成! 结果保存为:")
    print("- gait_constraint_validation.png")
    print("- phase_coupling_analysis.png")