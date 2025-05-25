import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import time
import glob
import torch
from model.SemLA import SemLA
import matplotlib.pyplot as plt
import numpy as np
from einops.einops import rearrange
from model.utils import YCbCr2RGB, RGB2YCrCb, make_matching_figure
import os
from ultralytics import YOLO


# 调整热成像视频的尺寸与红外视频一致
def resize_frame(frame, target_width, target_height):
    return cv2.resize(frame, (target_width, target_height))


def replace_zeroes(data):
    """ 避免 log(0) 计算错误，保证数据最小为 1 """
    return np.clip(data, 1, None)


def MSRCR(img, scales):
    img = img.astype(np.float32)
    img = replace_zeroes(img)
    log_img = np.log(img)

    def process_scale(scale):
        blurred = cv2.blur(img, (scale, scale))
        blurred = replace_zeroes(blurred)
        return log_img - cv2.log(blurred)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_scale, s) for s in scales]
        results = [f.result() for f in futures]

    MSRCR = sum(results) / len(results)
    MSRCR = np.exp(MSRCR)

    # 自适应拉伸
    # lower, upper = np.percentile(MSRCR, [0.01, 99.99])
    lower, upper = np.percentile(MSRCR, [0.2, 99.9])
    MSRCR = np.clip(MSRCR, lower, upper)
    MSRCR = cv2.normalize(MSRCR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return MSRCR

def SemLA_fusion(dehazed_infrared_frame, thermal_frame_resized,matcher,match_mode):
    # 读取可见光图像并转换为RGB格式
    # img0_raw = cv2.imread(img0_pth)
    img0_raw = cv2.cvtColor(dehazed_infrared_frame, cv2.COLOR_BGR2RGB)
    # 读取红外图像为灰度图
    # img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.cvtColor(thermal_frame_resized, cv2.COLOR_BGR2GRAY)
    # 调整图像大小为320x240（输入尺寸应该能被8整除）
    img0_raw = cv2.resize(img0_raw, (320, 240))
    img1_raw = cv2.resize(img1_raw, (320, 240))

    # 将图像转换为PyTorch张量并归一化，然后移至GPU
    img0 = torch.from_numpy(img0_raw)[None].cuda() / 255.  # 可见光图像张量
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.  # 红外图像张量

    # 重排可见光图像张量的维度顺序为[batch, channel, height, width]
    img0 = rearrange(img0, 'n h w c ->  n c h w')
    # 将RGB图像转换为YCbCr颜色空间
    vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img0)

    # 使用SemLA模型进行特征匹配
    mkpts0, mkpts1, feat_sa_vi, feat_sa_ir, sa_ir = matcher(vi_Y, img1, matchmode=match_mode)
    # 将匹配点转换为NumPy数组
    mkpts0 = mkpts0.cpu().numpy()  # 可见光图像中的匹配点
    mkpts1 = mkpts1.cpu().numpy()  # 红外图像中的匹配点

    # 使用RANSAC算法找到单应性矩阵，用于过滤异常匹配点
    _, prediction = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5)
    # 将预测结果转换为布尔数组
    prediction = np.array(prediction, dtype=bool).reshape([-1])
    # 筛选出符合单应性变换的匹配点
    mkpts0_tps = mkpts0[prediction]  # 筛选后的可见光图像匹配点
    mkpts1_tps = mkpts1[prediction]  # 筛选后的红外图像匹配点
    # 创建薄板样条变换器
    tps = cv2.createThinPlateSplineShapeTransformer()
    # 重塑匹配点数组以适应TPS变换器的输入格式
    mkpts0_tps = mkpts0_tps.reshape(1, -1, 2)
    mkpts1_tps = mkpts1_tps.reshape(1, -1, 2)

    # 创建匹配对象列表
    matches = []
    for j in range(1, mkpts0.shape[0] + 1):
        matches.append(cv2.DMatch(j, j, 0))

    # 估计TPS变换
    tps.estimateTransformation(mkpts0_tps, mkpts1_tps, matches)
    # 对红外图像应用TPS变换
    img1_raw_trans = tps.warpImage(img1_raw)
    # 对特征图应用相同的TPS变换
    sa_ir = tps.warpImage(sa_ir[0][0].detach().cpu().numpy())
    # 将变换后的特征图转换回PyTorch张量并移至GPU
    sa_ir = torch.from_numpy(sa_ir)[None][None].cuda()

    # 将变换后的红外图像转换为PyTorch张量并归一化
    img1_trans = torch.from_numpy(img1_raw_trans)[None][None].cuda() / 255.
    # 使用融合模型生成融合图像
    fuse = matcher.fusion(torch.cat((vi_Y, img1_trans), dim=0), sa_ir, matchmode=match_mode).detach()

    # 将融合结果从YCbCr转换回RGB颜色空间
    fuse = YCbCr2RGB(fuse, vi_Cb, vi_Cr)
    # 处理融合结果张量
    fuse = fuse.detach().cpu()[0]
    # 重排融合结果的维度顺序为[height, width, channel]
    fuse = rearrange(fuse, ' c h w ->  h w c').detach().cpu().numpy()
    
    # 打印融合图像的大小
    # print(f"融合图像大小: 高度={fuse.shape[0]}像素, 宽度={fuse.shape[1]}像素, 通道数={fuse.shape[2]}")
        
    # 将融合图像从RGB转换为BGR（OpenCV格式）
    fuse_bgr = cv2.cvtColor(np.clip(fuse * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # 将融合图像调整为原始图像大小
    original_height, original_width = dehazed_infrared_frame.shape[:2]
    fuse_bgr_resized = cv2.resize(fuse_bgr, (original_width, original_height))
    
    # 打印调整后的图像大小
    print(f"调整后的图像大小: 高度={fuse_bgr_resized.shape[0]}像素, 宽度={fuse_bgr_resized.shape[1]}像素")
    
    return fuse_bgr_resized
    # output_path = os.path.join(result_path, image_name)
    # cv2.imwrite(output_path, fuse_bgr_resized)
    
    # print(f"已保存融合图像: {output_path}")


if __name__ == "__main__":
    reg_weight_path = "/root/autodl-tmp/SemLA/pre-train/reg.ckpt"  # 配准模型权重路径
    fusion_weight_path = "/root/autodl-tmp/SemLA/pre-train/fusion75epoch.ckpt"  # 融合模型权重路径
    match_mode = 'semantic'  # 匹配模式，可选'semantic'或'scene'
    yolo_model=YOLO("/root/autodl-tmp/yolov8n.pt")

    # 加载视频文件
    infrared_video_path = '/root/autodl-tmp/videoDuiQi/1/output_rgb_smoked.mp4'  # 红外视频路径
    thermal_video_path = '/root/autodl-tmp/videoDuiQi/1/output_tr_smoked.mp4'  # 热成像视频路径

    # 创建输出目录
    output_dir = 'output_videos'
    # 确保结果保存目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化SemLA模型
    matcher = SemLA()
    # 加载配准模型的权重
    matcher.load_state_dict(torch.load(reg_weight_path), strict=False)
    # 加载融合模型的权重
    matcher.load_state_dict(torch.load(fusion_weight_path), strict=False)
    # 将模型设置为评估模式并移至GPU
    matcher = matcher.eval().cuda()

    # scales = [15, 105, 355, 755]
    scales = [15, 105, 775]
    # 设置滤波参数
    radius = 40  # 窗口半径
    eps = 1  # 正则化参数（较小保留更多细节，较大更平滑）

    infrared_video = cv2.VideoCapture(infrared_video_path)
    thermal_video = cv2.VideoCapture(thermal_video_path)

    if not infrared_video.isOpened():
        print("无法打开红外视频！")
        exit()
    if not thermal_video.isOpened():
        print("无法打开热成像视频！")
        exit()

    # 获取红外视频的帧率和尺寸
    infrared_fps = infrared_video.get(cv2.CAP_PROP_FPS)  # 25fps
    infrared_width = int(infrared_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    infrared_height = int(infrared_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取热成像视频的帧率和尺寸
    thermal_fps = thermal_video.get(cv2.CAP_PROP_FPS)  # 30fps
    thermal_width = int(thermal_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    thermal_height = int(thermal_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置输出视频的帧率为25fps，以适配红外视频的帧率
    output_fps = infrared_fps  # 使用红外视频的帧率

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    infrared_output_path = os.path.join(output_dir, 'dehazed_infrared.mp4')
    thermal_output_path = os.path.join(output_dir, 'resized_thermal.mp4')
    fusion_output_path = os.path.join(output_dir, 'fusion_result0.mp4')  # 新增融合结果视频路径

    infrared_writer = cv2.VideoWriter(infrared_output_path, fourcc, output_fps, (infrared_width, infrared_height))
    thermal_writer = cv2.VideoWriter(thermal_output_path, fourcc, output_fps, (infrared_width, infrared_height))
    fusion_writer = cv2.VideoWriter(fusion_output_path, fourcc, output_fps, (infrared_width, infrared_height))  # 新增融合结果视频写入器

    if not infrared_writer.isOpened():
        print("红外视频写入器初始化失败！")
    if not thermal_writer.isOpened():
        print("热成像视频写入器初始化失败！")
    if not fusion_writer.isOpened():
        print("融合结果视频写入器初始化失败！")

    # 获取视频总帧数
    infrared_frame_count = int(infrared_video.get(cv2.CAP_PROP_FRAME_COUNT))
    thermal_frame_count = int(thermal_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 读取视频帧并进行处理
    infrared_frame_idx = 0
    thermal_frame_idx = 0
    last_infrared_time = 0  # 红外视频的最后时间戳
    
    # 添加帧率统计相关变量
    start_time = time.time()
    last_fps_print_time = start_time
    frames_since_last_print = 0
    total_frames_processed = 0

    # 对齐视频帧并输出
    while infrared_frame_idx < infrared_frame_count:
        # 获取红外视频的当前时间戳（以秒为单位）

        infrared_time = infrared_frame_idx / infrared_fps

        # 读取红外视频的当前帧
        ret_infrared, infrared_frame = infrared_video.read()
        if not ret_infrared:
            print("ret_infrared--break")
            break

        # 读取热成像视频的对应帧
        if thermal_frame_idx < thermal_frame_count:
            thermal_video.set(cv2.CAP_PROP_POS_FRAMES, thermal_frame_idx)
            ret_thermal, thermal_frame = thermal_video.read()
            if not ret_thermal:
                print("ret_thermal--break")
                break
        else:
            print("break")
            break

        # 通过时间戳来同步热成像视频的帧
        if infrared_time > last_infrared_time:
            thermal_frame_idx = int(thermal_fps * infrared_time)

        # 调整热成像帧的大小与红外视频一致
        thermal_frame_resized = resize_frame(thermal_frame, infrared_width, infrared_height)

        # 对红外图像进行去雾
        gray_infrared_frame = cv2.cvtColor(infrared_frame, cv2.COLOR_BGR2GRAY)
        gray_dehazed_infrared_frame = MSRCR(gray_infrared_frame, scales)

        # 引导滤波
        gray_dehazed_infrared_frame = cv2.ximgproc.guidedFilter(guide=gray_infrared_frame, src=gray_dehazed_infrared_frame,
                                                                radius=radius, eps=eps, dDepth=-1)
        dehazed_infrared_frame = cv2.cvtColor(gray_dehazed_infrared_frame, cv2.COLOR_GRAY2BGR)

        # 使用SemLA进行图像融合
        fusion_frame = SemLA_fusion(dehazed_infrared_frame, thermal_frame_resized, matcher, match_mode)
        #yolo进行目标检测
        result=yolo_model(fusion_frame)
        fusion_frame_yolo=result[0].plot()
        # 写入融合结果
        fusion_writer.write(fusion_frame_yolo)

         # 更新帧索引
        infrared_frame_idx += 1
        last_infrared_time = infrared_time
        
        # 更新帧率统计
        frames_since_last_print += 1
        total_frames_processed += 1
        
        # 每秒打印一次帧率
        current_time = time.time()
        elapsed_since_last_print = current_time - last_fps_print_time
        
        if elapsed_since_last_print >= 1.0:  # 每秒打印一次
            fps = frames_since_last_print / elapsed_since_last_print
            total_elapsed = current_time - start_time
            avg_fps = total_frames_processed / total_elapsed if total_elapsed > 0 else 0
            progress_percent = (total_frames_processed / infrared_frame_count) * 100
            
            print(f"处理进度: {total_frames_processed}/{infrared_frame_count} 帧 ({progress_percent:.1f}%) - 当前帧率: {fps:.2f} fps - 平均帧率: {avg_fps:.2f} fps")
            
            # 重置计数器
            frames_since_last_print = 0
            last_fps_print_time = current_time
        print(f"已处理并融合第 {infrared_frame_idx}/{infrared_frame_count} 帧")
        
        # 写入视频帧
        # infrared_writer.write(dehazed_infrared_frame)
        # thermal_writer.write(thermal_frame_resized)

        #更新帧索引
        #infrared_frame_idx += 1
        #last_infrared_time = infrared_time

        # print(f"红外视频读取结束（已处理 {infrared_frame_idx}/{infrared_frame_count} 帧）")
    # 计算总处理时间和平均帧率
    total_time = time.time() - start_time
    final_avg_fps = total_frames_processed / total_time if total_time > 0 else 0

    # 释放资源
    infrared_video.release()
    thermal_video.release()
    infrared_writer.release()
    thermal_writer.release()
    fusion_writer.release()  # 释放融合结果视频写入器

    print("\n处理完成！")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"总处理帧数: {total_frames_processed} 帧")
    print(f"平均处理帧率: {final_avg_fps:.2f} fps")
    # print(f"去雾后的红外视频保存在: {infrared_output_path}")
    # print(f"调整尺寸后的热成像视频保存在: {thermal_output_path}")
    print(f"融合结果视频保存在: {fusion_output_path}")

