from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import io
from PIL import Image


# 파일 경로를 설정합니다.
file_path = 'models/underground_bts_eigen_v2_pytorch_retrain2/summaries/events.out.tfevents.1697114581.air'

# EventAccumulator를 초기화하고 파일을 로드합니다.
ea = event_accumulator.EventAccumulator(file_path)
ea.Reload()  # 로그 데이터를 불러옵니다.
print(ea.Tags())
# 사용 가능한 모든 지표를 확인합니다.
scalars = ea.Tags()['scalars']

for scalar in scalars:
    if scalar == 'silog_loss':  # 'var_average'를 제외합니다.
        values = ea.Scalars(scalar)
        x = [v.step for v in values]
        y = [v.value for v in values]
        
        plt.plot(x, y, label=scalar)

plt.xlabel('Step')
plt.ylabel('Value')
plt.legend()
plt.show()


# 사용 가능한 모든 이미지를 확인합니다.
images = ea.Tags()['images']

# 모든 이미지를 순회하며 그래프를 그립니다.
for image_tag in images:
    image_event = ea.Images(image_tag)
    
    # 이미지 이벤트에서 이미지를 로딩합니다.
    for img in image_event:
        image_stream = io.BytesIO(img.encoded_image_string)
        image = Image.open(image_stream)
        
        # 이미지를 시각화합니다.
        plt.imshow(image)
        plt.title(f"Step {img.step}: {image_tag}")
        plt.axis('off')
        plt.show()