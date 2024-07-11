import os

base_folder = "/home/eunseong/catkin_ws/src/BTS_ws/dataset/underground/240611"
# cam_folders = ["4cam_3","4cam_4", "4cam_5", "4cam_6"]
cam_folders = ["UNDER5","UNDER4", "UNDER3"]

output_file_path = "underground_eigen_train_files_with_gt_under240611.txt"

with open(output_file_path, 'w') as file:
    for cam_folder in cam_folders:
        # 각 폴더에 대해 경로 설정
        color_folder = os.path.join(base_folder, cam_folder, "color")
        depth_folder = os.path.join(base_folder, cam_folder, "depth")

        # 이미지 파일을 찾음
        color_images = sorted([f for f in os.listdir(color_folder) if f.endswith('.png')])
        depth_images = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')])

        # 모든 이미지 파일을 순회
        for color_img, depth_img in zip(color_images, depth_images):
            # 파일에 경로 및 숫자를 작성
            file.write(f"{cam_folder}/color/{color_img} {cam_folder}/depth/{depth_img} 298.3504\n")
