import torch

# 模拟设备（例如 'cpu' 或 'cuda'）
device = 'cpu'

# 假设 self.Np 是某个预定义的数量
self = type('self', (object,), {'Np': 4})()  # 示例中 Np 设置为 4

# 初始化 box_token_mask 和 last_hidden_state
batch_size, sequence_length = 1, 752
box_token_mask = torch.rand(batch_size, sequence_length, device=device) > 0.5  # 随机生成布尔掩码

# 打印原始的 box_token_mask 形状
print("Original box_token_mask shape:", box_token_mask.shape)

# 应用给定的拼接操作
# 在末尾添加一列 False
box_token_mask = torch.cat([
    box_token_mask,
    torch.zeros(box_token_mask.shape[0], 1, dtype=torch.bool, device=device)
], dim=1)

# 在开头添加 Np - 1 列 False
box_token_mask = torch.cat([
    torch.zeros(box_token_mask.shape[0], self.Np - 1, dtype=torch.bool, device=device),
    box_token_mask
], dim=1)

# 打印修改后的 box_token_mask 形状
print("Modified box_token_mask shape:", box_token_mask.shape)

# 初始化 last_hidden_state
hidden_size = 3584
sequence_length_modified = box_token_mask.shape[1]  # 修改后的序列长度应与 box_token_mask 相匹配
last_hidden_state = torch.randn(batch_size, sequence_length_modified, hidden_size, device=device)

# 获取有效的索引（即 mask 为 True 的位置）
valid_indices = torch.nonzero(box_token_mask).squeeze()

# 提取框特征
if valid_indices.dim() == 1:  # 如果只有一个非零元素，则需要调整维度
    valid_indices = valid_indices.unsqueeze(0)

box_features = last_hidden_state[valid_indices[:, 0], valid_indices[:, 1]]

print("Box Features Shape:", box_features.shape)
print("Box Features:\n", box_features)