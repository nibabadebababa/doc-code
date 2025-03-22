# from transformers import Qwen2TokenizerFast

# tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
# tokenizer("Hello world")["input_ids"]
# print(tokenizer("Hello world"))
if __name__ == "__main__":
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from qwen_vl_utils import process_vision_info

    # 设置路径
    base_image_path = "/root/autodl-tmp/DocTamper/image/image_10023.jpg"

    # 加载 tokenizer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # 加载模型
    model = AutoModelForVision2Seq.from_pretrained("/root/autodl-tmp/models/Qwen2-VL-7B-Instruct")

    # 获取 Vision Tower
    vision_tower = model.vision_tower

    # 构造对话输入，符合 Qwen2-VL 视觉处理要求
    conversations = [
        {
            "role": "system",
            "content": "You are an AI assistant capable of understanding images."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "value": base_image_path},
                {"type": "text", "value": "What is in the image?"}
            ]
        }
    ]

    # 处理视觉信息
    vision_inputs = process_vision_info(conversations)

    # 视觉编码器提取特征
    with torch.no_grad():
        image_features = vision_tower(vision_inputs)

    print("Extracted Vision Features:", image_features.shape)


