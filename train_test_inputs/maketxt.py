import os

base_folder = "/home/eunseong/catkin_ws/src/BTS_ws/dataset/underground/240611"
# cam_folders = ["UNDER5", "UNDER4", "UNDER3"]
cam_folders = ["UNDER2"]

output_file_path = "underground_eigen_test_files_with_gt_under240611.txt"

# 각 카메라 각도에 따른 초점 거리
focal_lengths = {
    "front_cam": 296.0549,
    "rear_cam": 297.9473,
    "left_cam": 296.7004,
    "right_cam": 294.9383
}

camera_angles = ["front_cam", "rear_cam", "left_cam", "right_cam"]

with open(output_file_path, 'w') as file:
    for cam_folder in cam_folders:
        # 각 카메라 각도에 대해 이미지 파일을 모음
        image_dict = {angle: sorted([f for f in os.listdir(os.path.join(base_folder, cam_folder, angle, "color")) if f.startswith('raw_image_') and f.endswith('.png')]) for angle in camera_angles}
        
        # 가장 작은 이미지 파일 수를 기준으로 순회
        min_length = min(len(image_dict[angle]) for angle in camera_angles)
        
        # 240711: 현재 카메라 별 sequence. 
        for angle in camera_angles:
            for i in range(min_length):
            # for angle in camera_angles:
                color_folder = os.path.join(base_folder, cam_folder, angle, "color")
                depth_folder = os.path.join(base_folder, cam_folder, angle, "depth")
                
                color_img = image_dict[angle][i]
                # depth 이미지 파일 이름은 color 이미지 파일 번호를 가져와서 이름을 형성
                depth_img_number = color_img.replace('raw_image_', '').replace('.png', '')
                depth_img = f"depth_image_{depth_img_number}.png"
                
                # 파일에 경로 및 초점 거리를 작성
                file.write(f"{cam_folder}/{angle}/color/{color_img} {cam_folder}/{angle}/depth/{depth_img} {focal_lengths[angle]}\n")
