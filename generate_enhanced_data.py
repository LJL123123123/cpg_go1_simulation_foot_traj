#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成具有增强耦合约束的足端轨迹数据
确保Walk模式同时只有一个足端腾空，Trot模式只有对角足端同时腾空
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cpg_go1_simulation.stein.foot_trajectory_cpg import FootTrajectoryCPG

def generate_enhanced_foot_trajectory_data():
    """生成增强耦合约束的足端轨迹数据"""
    
    # 测试参数
    gait_types = {
        1: "walk",
        2: "trot", 
        3: "pace",
        4: "bound",
        5: "pronk"
    }
    
    duration = 5.0  # 每个步态测试5秒
    
    print("=== 生成增强耦合约束的足端轨迹数据 ===")
    
    for gait_id, gait_name in gait_types.items():
        print(f"\n生成 {gait_name} 步态数据...")
        
        # 初始化CPG
        cpg = FootTrajectoryCPG(
            before_ftype=gait_id,
            after_ftype=gait_id,
            total_time=duration,
            toc=duration + 1.0  # 不触发转换
        )
        
        # 导出数据
        print(f"  导出CSV数据...")
        cpg.export_csv()
        
        # 生成可视化
        print(f"  生成可视化图表...")
        cpg.plot_foot_trajectories(
            save_path=f'/home/cpg_go1_simulation/enhanced_{gait_name}_trajectory.png'
        )
        
        # 验证约束条件
        print(f"  验证步态约束...")
        validate_gait_constraints(cpg, gait_id, gait_name, duration)

def validate_gait_constraints(cpg, gait_id, gait_name, duration):
    """验证步态约束条件"""
    
    dt = 0.02  # 50Hz采样
    time_steps = int(duration / dt)
    times = np.linspace(0, duration, time_steps)
    
    violations = 0
    airborne_counts = []
    
    for t in times:
        # 获取所有足端相位信息
        foot_phases = cpg.get_all_foot_phases(t)
        foot_phases = cpg.enforce_gait_constraints(foot_phases, "LF")
        
        # 统计腾空足端
        airborne_feet = []
        for foot_name in cpg.foot_names:
            if not foot_phases[foot_name]['is_stance']:
                airborne_feet.append(foot_name)
        
        num_airborne = len(airborne_feet)
        airborne_counts.append(num_airborne)
        
        # 检查约束违反
        constraint_violation = False
        
        if gait_id == 1:  # Walk: 最多1个足端腾空
            constraint_violation = num_airborne > 1
        elif gait_id == 2:  # Trot: 最多2个足端腾空，且必须是对角
            if num_airborne > 2:
                constraint_violation = True
            elif num_airborne == 2:
                # 检查是否是对角对
                diagonal_pairs = [("LF", "RH"), ("RF", "LH")]
                is_diagonal = any(
                    set(airborne_feet) == set(pair) for pair in diagonal_pairs
                )
                if not is_diagonal:
                    constraint_violation = True
        
        if constraint_violation:
            violations += 1
    
    # 输出统计结果
    violation_rate = violations / len(times) * 100
    avg_airborne = np.mean(airborne_counts)
    max_airborne = max(airborne_counts)
    
    print(f"    约束违反率: {violation_rate:.2f}%")
    print(f"    平均腾空足端数: {avg_airborne:.2f}")
    print(f"    最大腾空足端数: {max_airborne}")
    
    if violation_rate == 0.0:
        print(f"    ✅ {gait_name}步态约束完全满足!")
    else:
        print(f"    ❌ {gait_name}步态约束有违反")

def compare_before_after_enhancement():
    """对比增强前后的步态特性"""
    
    print("\n=== 对比增强前后的步态特性 ===")
    
    # 重点测试Walk和Trot
    test_gaits = [(1, "Walk"), (2, "Trot")]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('增强耦合约束前后对比', fontsize=16)
    
    for gait_idx, (gait_id, gait_name) in enumerate(test_gaits):
        
        # 初始化CPG
        cpg = FootTrajectoryCPG(
            before_ftype=gait_id,
            after_ftype=gait_id,
            total_time=3.0,
            toc=4.0
        )
        
        duration = 3.0
        dt = 0.01
        times = np.arange(0, duration, dt)
        
        # 记录轨迹数据
        trajectories = {name: [] for name in cpg.foot_names}
        airborne_states = []
        
        for t in times:
            # 获取增强约束后的足端位置
            current_airborne = 0
            for foot_name in cpg.foot_names:
                pos = cpg.generate_foot_position(foot_name, t)
                trajectories[foot_name].append(pos)
                
                # 检查是否腾空 (z坐标 > 基础位置)
                base_z = cpg.foot_base_positions[foot_name][2]
                if pos[2] > base_z + 0.01:  # 1cm以上认为腾空
                    current_airborne += 1
            
            airborne_states.append(current_airborne)
        
        # 绘制3D轨迹
        ax1 = fig.add_subplot(2, 2, gait_idx*2 + 1, projection='3d')
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, foot_name in enumerate(cpg.foot_names):
            traj = np.array(trajectories[foot_name])
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    color=colors[i], label=foot_name, linewidth=2)
        
        ax1.set_title(f'{gait_name} - 3D足端轨迹')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        
        # 绘制腾空足端数量
        ax2 = axes[gait_idx, 1]
        ax2.plot(times, airborne_states, 'b-', linewidth=2)
        ax2.set_title(f'{gait_name} - 腾空足端数量')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('腾空足端数量')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.5, 4.5)
        
        # 添加约束线
        if gait_id == 1:  # Walk
            ax2.axhline(y=1, color='g', linestyle='--', alpha=0.7, 
                       label='最大允许(1)')
        elif gait_id == 2:  # Trot  
            ax2.axhline(y=2, color='g', linestyle='--', alpha=0.7,
                       label='最大允许(2)')
        ax2.legend()
        
        # 输出统计
        avg_airborne = np.mean(airborne_states)
        max_airborne = max(airborne_states)
        print(f"\n{gait_name}步态统计:")
        print(f"  平均腾空足端数: {avg_airborne:.2f}")
        print(f"  最大腾空足端数: {max_airborne}")
        
        # 验证约束
        if gait_id == 1 and max_airborne <= 1:
            print(f"  ✅ Walk约束满足 (最多1个腾空)")
        elif gait_id == 2 and max_airborne <= 2:
            print(f"  ✅ Trot约束满足 (最多2个腾空)")
        else:
            print(f"  ❌ 约束违反")
    
    plt.tight_layout()
    plt.savefig('/home/cpg_go1_simulation/enhanced_coupling_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    # 1. 生成所有步态的增强轨迹数据
    generate_enhanced_foot_trajectory_data()
    
    # 2. 对比增强前后的效果
    compare_before_after_enhancement()
    
    print("\n=== 增强耦合约束足端轨迹数据生成完成 ===")
    print("\n生成的文件:")
    print("- enhanced_walk_trajectory.png")
    print("- enhanced_trot_trajectory.png") 
    print("- enhanced_pace_trajectory.png")
    print("- enhanced_bound_trajectory.png")
    print("- enhanced_pronk_trajectory.png")
    print("- enhanced_coupling_comparison.png")
    print("- 各种CSV数据文件")