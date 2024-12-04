import onnxruntime as rt
import os
import argparse
from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer import optimize
import onnx
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_images(image_paths, input_size, mean, std):
    images = []
    for path in image_paths:
        # 读取图像
        img = Image.open(path).convert('RGB')  # 确保图像为 RGB 模式
        new_img_width = input_size[1]
        new_img_height = float(new_img_width)/img.size[0] * img.size[1]
        
        #Resize
        img_resized = img.resize((int(new_img_width), int(new_img_height)), Image.LANCZOS)
        crop_x = (img_resized.size[0] - input_size[1]) / 2
        crop_y = (img_resized.size[1] - input_size[0]) / 2

        crop_img = img_resized.crop((crop_x, crop_y, crop_x+input_size[1], crop_y+input_size[0]))
        assert crop_img.size[0] == input_size[1] and crop_img.size[1] == input_size[0]
        # 缩放
        #img_resized = img.resize((input_size[1], input_size[0]), Image.BICUBIC)
        
        # 转换为 numpy 数组并归一化
        img_array = np.array(crop_img)
        
        # 减均值，除以标准差
        img_normalized = ((img_array - mean) / std)
        
        # 通道转换（HWC -> CHW）
        img_transposed = img_normalized.transpose(2, 0, 1)
        
        images.append(img_transposed)
    
    # 将图像堆叠为批量输入
    batch_images = np.stack(images, axis=0)
    return batch_images

so = rt.SessionOptions()
so.log_severity_level = 3
so.intra_op_num_threads = os.cpu_count()
so.inter_op_num_threads = os.cpu_count()


parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    default=[],
    help="Model name to be added to the list to run",
)

parser.add_argument(
    "-a",
    "--artifacts",
    default=[],
    help="Artifacts folder to store the compiled model",
)

parser.add_argument(
    "-p",
    "--prefix",
    default=[],
    help="Prefix to add to the image paths",
)

args = parser.parse_args()

required_options = {
    "tidl_tools_path":  os.environ["TIDL_TOOLS_PATH"],
    "artifacts_folder": args.artifacts,
}

#Delete the artifacts folder if it exists
if os.path.exists(required_options["artifacts_folder"]):
    shutil.rmtree(required_options["artifacts_folder"])

os.makedirs(required_options["artifacts_folder"], exist_ok=True)


delegate_options = {}
delegate_options.update(required_options)
delegate_options["advanced_options:calibration_frames"] = 1


copy_path = args.model[:-5] + "_opt.onnx"
# Check if copy path exists and prompt for permission to overwrite
shutil.copy2(args.model, copy_path)
print(
    f"\033[93mOptimization Enabled: Moving {args.model} to {copy_path} before overwriting by optimization\033[00m"
)
args.model = copy_path
optimize(args.model, args.model)

onnx.shape_inference.infer_shapes_path(args.model, args.model)

#Parsing the model
EP_list = ["TIDLCompilationProvider", "CPUExecutionProvider"]
sess = rt.InferenceSession(
            args.model,
            providers=EP_list,
            provider_options=[delegate_options, {}],
            sess_options=so,
)

#Calibration
# 假设您的模型期望输入尺寸为 256x704
input_size = (256, 704)  # (height, width)

# 使用与训练时相同的均值和标准差
mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

image_paths = [
    '0-FRONT.jpg',
    '1-FRONT_RIGHT.jpg',
    '2-FRONT_LEFT.jpg',
    '3-BACK.jpg',
    '4-BACK_LEFT.jpg',
    '5-BACK_RIGHT.jpg',
    # ... 添加其他相机的图像路径
]

image_paths = [os.path.join(args.prefix, path) for path in image_paths]

batch_images = preprocess_images(image_paths, input_size, mean, std)

fig, axes = plt.subplots(1, len(batch_images), figsize=(20, 5))
for i, ax in enumerate(axes):
    img = batch_images[i].transpose(1, 2, 0)*std + mean  # 恢复原始图像
    img = np.clip(img, 0, 255).astype(np.uint8)  # 限制值范围并转换为 uint8 类型
    ax.imshow(img)
    ax.axis('off')
plt.savefig('input_images.png')

backbone_input_name = sess.get_inputs()[0].name
backbone_output_name = sess.get_outputs()[0].name
batch_size = batch_images.shape[0]
num_cameras = batch_images.shape[0]  # 假设每个图像对应一个相机
channels = batch_images.shape[1]
height = batch_images.shape[2]
width = batch_images.shape[3]

backbone_input = batch_images.reshape(1, num_cameras, channels, height, width)


print(f"Running inference on {args.model} with input shape {backbone_input.shape}")
backbone_output = sess.run(
    [backbone_output_name],
    {backbone_input_name: backbone_input}
)[0]
print(f"Output shape: {backbone_output.shape}")

num_images = backbone_output.shape[0]
num_channels = 16  # 只显示前 16 个通道

grid_rows = num_images
grid_cols = num_channels

fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 1.5, grid_rows * 1.5))

for img_idx in range(num_images):
    for ch_idx in range(num_channels):
        ax = axes[img_idx, ch_idx]
        feature_map = backbone_output[img_idx, ch_idx, :, :]
        ax.imshow(feature_map, cmap='viridis')
        ax.axis('off')

plt.tight_layout()
plt.savefig('feature_maps.png')
    
