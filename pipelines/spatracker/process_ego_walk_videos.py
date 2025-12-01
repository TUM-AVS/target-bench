#!/usr/bin/env python3
"""
视频处理脚本：对EgoWalk_samples中的视频进行修复、裁剪、拼接等操作
参照world-decoder_code_spatracker/combine_video.sh的格式
"""

import os
import json
import subprocess
import glob
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, cwd=None, timeout=300):
    """运行shell命令并返回结果"""
    try:
        logger.info(f"执行命令: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        if result.returncode != 0:
            logger.error(f"命令执行失败: {command}")
            logger.error(f"错误输出: {result.stderr}")
            return False, result.stderr
        logger.info(f"命令执行成功")
        return True, result.stdout
    except subprocess.TimeoutExpired:
        logger.error(f"命令执行超时 ({timeout}秒): {command}")
        return False, f"Command timed out after {timeout} seconds"
    except Exception as e:
        logger.error(f"命令执行异常: {e}")
        return False, str(e)

def get_frame_count(video_path):
    """获取视频帧数"""
    command = f"ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 '{video_path}'"
    success, output = run_command(command)
    if success and output.strip():
        return int(output.strip())
    else:
        logger.error(f"无法获取视频帧数: {video_path}")
        return 0

def process_single_folder(folder_path):
    """处理单个文件夹中的视频"""
    folder_name = os.path.basename(folder_path)
    logger.info(f"处理文件夹: {folder_name}")
    
    # 查找视频文件
    video_2024_pattern = os.path.join(folder_path, "sample*_video.mp4")
    video_2024_files = glob.glob(video_2024_pattern)
    
    current_frame_pattern = os.path.join(folder_path, "*_current_frame.mp4")
    current_frame_files = glob.glob(current_frame_pattern)
    
    # 查找metadata文件
    metadata_pattern = os.path.join(folder_path, "*_metadata.json")
    metadata_files = glob.glob(metadata_pattern)
    
    if not video_2024_files:
        logger.warning(f"未找到2024*_video.mp4格式的视频文件: {folder_path}")
        return None
    
    if not current_frame_files:
        logger.warning(f"未找到*_current_frame.mp4格式的视频文件: {folder_path}")
        return None
    
    if not metadata_files:
        logger.warning(f"未找到metadata.json文件: {folder_path}")
        return None
    
    # 使用第一个找到的文件
    video_2024 = video_2024_files[0]
    current_frame_video = current_frame_files[0]
    metadata_file = metadata_files[0]
    
    logger.info(f"使用视频文件: {os.path.basename(video_2024)}")
    logger.info(f"使用current_frame视频: {os.path.basename(current_frame_video)}")
    logger.info(f"使用metadata: {os.path.basename(metadata_file)}")
    
    # 读取metadata文件
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"读取metadata文件失败: {e}")
        return None
    
    # 提取时间戳信息
    try:
        first_frame = data['distance_first_to_current']["first_frame"]["trajectory_frame"]
        current_frame = data['distance_first_to_current']["current_frame"]["trajectory_frame"]
        
        # 计算分割点时间戳 (原始视频帧率为5fps)
        fps = 5
        duration = (current_frame - first_frame) / fps
        frames_to_keep = current_frame - first_frame
        
        logger.info(f"First frame: {first_frame}, Current frame: {current_frame}")
        logger.info(f"计算得到的分割时间: {duration}秒")
        logger.info(f"保留帧数: {frames_to_keep}")
        
    except KeyError as e:
        logger.error(f"metadata文件中缺少必要字段: {e}")
        return None
    
    # 步骤1: 修复视频 (使用libx264和aac编码)
    gt_fixed_path = os.path.join(folder_path, "gt_fixed.mp4")
    command1 = f"ffmpeg -y -loglevel error -i '{video_2024}' -c:v libx264 -c:a aac '{gt_fixed_path}'"
    success, _ = run_command(command1)
    if not success:
        logger.error(f"视频修复失败: {folder_name}")
        return None
    
    # 步骤2: 裁剪视频到指定时间
    first_to_current_path = os.path.join(folder_path, "first_current.mp4")
    command2 = f"ffmpeg -y -loglevel error -i '{gt_fixed_path}' -vf \"select='between(n\\,0\\,{frames_to_keep})'\" -vsync vfr '{first_to_current_path}'"
    success, _ = run_command(command2)
    if not success:
        logger.error(f"视频裁剪失败: {folder_name}")
        return None
    
    # 步骤3: 修改分辨率和帧率
    first_fixed_path = os.path.join(folder_path, "first_current_fixed.mp4")
    command3 = f"ffmpeg -y -loglevel error -i '{first_to_current_path}' -vf 'scale=1280:720,fps=30' -c:a copy '{first_fixed_path}'"
    success, _ = run_command(command3)
    if not success:
        logger.error(f"视频分辨率帧率修改失败: {folder_name}")
        return None
    
    # 步骤4: 创建视频列表文件并拼接
    list_file = os.path.join(folder_path, "list.txt")
    with open(list_file, 'w') as f:
        f.write("file 'first_current_fixed.mp4'\n")
        f.write(f"file '{os.path.basename(current_frame_video)}'\n")
    
    combined_path = os.path.join(folder_path, "combined.mp4")
    command4 = f"ffmpeg -y -loglevel error -f concat -safe 0 -i '{list_file}' -c copy '{combined_path}'"
    success, _ = run_command(command4)
    if not success:
        logger.error(f"视频拼接失败: {folder_name}")
        return None
    
    # 步骤5: 获取各视频的帧数
    first_fixed_frames = get_frame_count(first_fixed_path)
    current_frame_frames = get_frame_count(current_frame_video)
    combined_frames = get_frame_count(combined_path)
    
    # 清理临时文件
    try:
        os.remove(gt_fixed_path)
        os.remove(first_to_current_path)
        os.remove(list_file)
    except:
        pass
    
    result = {
        "folder_name": folder_name,
        "first_fixed_frames": first_fixed_frames,
        "current_frame_frames": current_frame_frames,
        "combined_frames": combined_frames,
        "duration_seconds": duration,
        "first_frame": first_frame,
        "current_frame": current_frame
    }
    
    logger.info(f"处理完成: {folder_name}")
    logger.info(f"First fixed frames: {first_fixed_frames}")
    logger.info(f"Current frame video frames: {current_frame_frames}")
    logger.info(f"Combined frames: {combined_frames}")
    
    return result

def main():
    """主函数"""
    # 设置路径
    base_path = "/home/hongyuan/world-decoder/dataset/Benchmark"
    
    if not os.path.exists(base_path):
        logger.error(f"基础路径不存在: {base_path}")
        return
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(base_path) 
                  if os.path.isdir(os.path.join(base_path, f))]
    
    if not subfolders:
        logger.warning("未找到任何子文件夹")
        return
    
    logger.info(f"找到 {len(subfolders)} 个子文件夹")
    
    # 处理每个子文件夹
    results = []
    for subfolder in subfolders:
        folder_path = os.path.join(base_path, subfolder)
        result = process_single_folder(folder_path)
        if result:
            results.append(result)
    
    # 保存结果到JSON文件
    output_file = "/home/hongyuan/world-decoder/video_processing_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"处理完成，结果已保存到: {output_file}")
    logger.info(f"成功处理 {len(results)} 个文件夹")
    
    # 打印汇总信息
    print("\n=== 处理结果汇总 ===")
    for result in results:
        print(f"\n文件夹: {result['folder_name']}")
        print(f"  first_current_fixed.mp4 帧数: {result['first_fixed_frames']}")
        print(f"  *_current_frame.mp4 帧数: {result['current_frame_frames']}")
        print(f"  combined.mp4 帧数: {result['combined_frames']}")
        print(f"  裁剪时长: {result['duration_seconds']:.2f}秒")
        print(f"  第一帧: {result['first_frame']}")
        print(f"  当前帧: {result['current_frame']}")

if __name__ == "__main__":
    main()
