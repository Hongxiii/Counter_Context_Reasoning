import json

# 读取 JSON 文件
with open('/data/lihongxi/Counter-Factual-Reasoning/results/pipeline_identification/sk-6H72K8DB8HOw0u8u1768Cc8e1fEa438284Ee38C807A7E76a-z-n-yes.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print(len(data))
def count_answer_starts(data):
    counts = {"t_No": 0, "t_Yes": 0, "t_NotSure": 0,
              "f_No": 0, "f_Yes": 0, "f_NotSure": 0}

    # 遍历所有键值对
    for key, value in data.items():
        # 只处理键中包含 "t" 的条目
        if 't' in key.lower():
            # 提取答案部分并转换为小写
            answer = value.lower()
            
            # 检查答案是否以 "no", "yes", 或 "notsure" 开头
            if answer.startswith("notsure"):
                counts["t_NotSure"] += 1
            elif "yes" in answer:
                counts["t_Yes"] += 1
            elif "no" in answer:
                counts["t_No"] += 1
            else:
                print(key,value)

        
        elif 'f' in key.lower():
            # 提取答案部分并转换为小写
            answer = value.lower()
            
            # 检查答案是否以 "no", "yes", 或 "notsure" 开头
            if answer.startswith("notsure"):
                counts["f_NotSure"] += 1
            elif "yes" in answer:
                counts["f_Yes"] += 1
            elif "no" in answer:
                counts["f_No"] += 1
            else:
                print(key,value)
    
    return counts

# 获取统计结果
counts = count_answer_starts(data)

# 打印结果
print(f"Keys containing 't' with answers starting with 'No': {counts['t_No']}")
print(f"Keys containing 't' with answers starting with 'Yes': {counts['t_Yes']}")
print(f"Keys containing 't' with answers starting with 'NotSure': {counts['t_NotSure']}")

print(f"Keys containing 'f' with answers starting with 'No': {counts['f_No']}")
print(f"Keys containing 'f' with answers starting with 'Yes': {counts['f_Yes']}")
print(f"Keys containing 'f' with answers starting with 'NotSure': {counts['f_NotSure']}")


TP = counts["t_Yes"]  # 正样本被正确分类
FN = counts["t_No"]   # 正样本被错误分类
FP = counts["f_Yes"]  # 负样本被错误分类

# 防止除以0的情况
if (TP + FP) > 0:
    precision = TP / (TP + FP)
else:
    precision = 0.0

if (TP + FN) > 0:
    recall = TP / (TP + FN)
else:
    recall = 0.0

if (precision + recall) > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0.0

print(f"p:{precision}")
print(f"r:{recall}")
print(f"f1:{f1_score}")
print((counts['t_Yes']+counts['f_No'])/sum(counts.values()))