import cv2
import os
import re

def make_colored_depth_video(image_folder, video_name, fps):
    """
    이미지 폴더에 있는 모든 이미지에 컬러맵을 적용하고, 이를 사용하여 비디오를 만듭니다.
    
    :param image_folder: 이미지 파일들이 저장된 폴더 경로
    :param video_name: 생성될 비디오 파일의 이름
    :param fps: 비디오의 프레임레이트 (frames per second)
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    
    # 파일 이름에서 숫자 부분을 추출하여 정렬
    images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    # print(images)
    
    # 첫 번째 이미지를 불러와서 사이즈를 얻습니다.
    example_img = cv2.imread(os.path.join(image_folder, images[0]), cv2.IMREAD_UNCHANGED)
    height, width = example_img.shape

    # 동영상 파일을 생성하기 위한 VideoWriter 객체를 설정합니다.
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for image_file in images:
        # 이미지를 불러옵니다.
        depth_image = cv2.imread(os.path.join(image_folder, image_file), cv2.IMREAD_UNCHANGED)
        
        # 이미지를 정규화합니다.
        normalized_depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 컬러맵을 적용합니다.
        colored_depth_image = cv2.applyColorMap(normalized_depth_image, cv2.COLORMAP_JET)
        
        # 비디오에 이미지 프레임을 추가합니다.
        video.write(colored_depth_image)
    
    # 비디오 파일을 저장합니다.
    video.release()

# 사용 예:
make_colored_depth_video('raw/', 'output_video.mp4', 30)
