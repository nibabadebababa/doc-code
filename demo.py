from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,VISION_START_TOKEN,VISION_END_TOKEN,LLAVA_IMAGE_TOKEN


# 模型路径
model_path = "/root/autodl-tmp/models/Qwen2-VL-7B-Instruct"

# 加载预训练模型和处理器
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_path)

# 构建消息
messages = [{"role": "user", "content": [
        {"type": "image", "image": "/root/autodl-tmp/DocTamper/image/image_10023.jpg"},
        {"type": "text", "text": "Describe this image."}
    ]}]


question = "hello?"
answer = "yes"

user_input = f"{DEFAULT_IM_START_TOKEN}user\n{question}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
gpt_response = f"{answer}{DEFAULT_IM_END_TOKEN}\n"

# 使用处理器应用聊天模板并获取文本输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Processed Text Input:\n", text)



# 处理视觉信息（图像和视频）
images, videos = process_vision_info(messages)

# 使用处理器准备输入张量
inputs = processor(text=user_input+gpt_response, images=images, videos=videos, padding=True, return_tensors="pt")
print("Input IDs:\n", inputs["input_ids"][0])
# 解码生成的 ID 为文本
inputs_text = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
# 打印生成的文本
print("Generated Output:\n", inputs_text)

# 将输入移动到 GPU（如果可用）
if torch.cuda.is_available():
    inputs = inputs.to('cuda')

# 生成输出
generated_ids = model.generate(**inputs)

# 解码生成的 ID 为文本
generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 打印生成的文本
print("Generated Output:\n", generated_text)