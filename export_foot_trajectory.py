#!/usr/bin/env python3
"""
足端轨迹CPG数据导出脚本
Export script for foot trajectory CPG data
"""

from cpg_go1_simulation.config import GAIT_MAP
from cpg_go1_simulation.stein.foot_trajectory_cpg import FootTrajectoryCPG


def export_foot_trajectory_data(
    gait_type: str = "walk",
    command_time: float = 15.0, 
    total_time: float = 25.0,
    _if_backward: bool = False,
    step_height: float = 0.08,
    step_length: float = 0.15,
    body_height: float = 0.25,
    foot_spacing: float = 0.2
):
    """
    生成足端轨迹CPG数据
    
    Args:
        gait_type (str): 步态类型，如 'walk', 'trot', 'pace', 'bound', 'pronk'
        command_time (float): 步态切换时间点
        total_time (float): 总仿真时间
        _if_backward (bool): 是否后退
        step_height (float): 抬腿高度 (米)
        step_length (float): 步长 (米)  
        body_height (float): 机身高度 (米)
        foot_spacing (float): 足端间距 (米)
    
    Returns:
        float: 实际转换时间 (如果有步态转换)
    """
    
    if "_to_" in gait_type:
        # 处理步态转换
        before_gait, after_gait = gait_type.split("_to_")
        before_gait_id = GAIT_MAP[before_gait]
        after_gait_id = GAIT_MAP[after_gait]
    else:
        gait_id = GAIT_MAP[gait_type]
        before_gait_id = after_gait_id = gait_id

    # 创建足端轨迹CPG实例
    foot_cpg = FootTrajectoryCPG(
        before_ftype=before_gait_id,
        after_ftype=after_gait_id,
        total_time=total_time,
        toc=command_time,
        _if_backward=_if_backward,
        step_height=step_height,
        step_length=step_length, 
        body_height=body_height,
        foot_spacing=foot_spacing
    )
    
    # 导出数据
    foot_cpg.export_csv()
    
    # 生成可视化
    plot_filename = f"foot_traj_{gait_type}_{total_time}s.png"
    foot_cpg.plot_foot_trajectories(plot_filename)
    
    return foot_cpg.toc if "_to_" in gait_type else None


def compare_gaits():
    """比较不同步态的足端轨迹"""
    
    print("生成不同步态的足端轨迹进行比较...")
    
    gaits = ["walk", "trot", "pace", "bound", "pronk"]
    
    for gait in gaits:
        print(f"\n生成 {gait} 步态的足端轨迹...")
        
        export_foot_trajectory_data(
            gait_type=gait,
            total_time=3.0,
            command_time=1.5
        )
        
    print("\n所有步态的足端轨迹数据已生成完成!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        gait = sys.argv[1]
        print(f"生成 {gait} 步态的足端轨迹...")
        export_foot_trajectory_data(gait_type=gait, total_time=3.0)
    else:
        # 默认生成walk步态
        print("生成默认walk步态的足端轨迹...")
        export_foot_trajectory_data(
            gait_type="walk", 
            total_time=3.0,
            step_height=0.08,
            step_length=0.15
        )
        
        # 可选：生成所有步态进行比较
        choice = input("\n是否要生成所有步态进行比较? (y/n): ").lower().strip()
        if choice == 'y':
            compare_gaits()