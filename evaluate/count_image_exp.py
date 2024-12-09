import json
import os

def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


# 获取文件夹下所有文件的路径

directory_path="/data/lihongxi/Counter-Factual-Reasoning/evaluate/image_explanation/"
all_file_paths = get_all_file_paths(directory_path)
# 将字符串解析为字典
for path in all_file_paths:
    class_dict = {'a': {}, 'b': {}, 'c': {}, 'd': {}, 'e': {}}
    with open(path, 'r') as file:
        data = json.load(file)
    for key, value in data.items():
        class_dict[key[0]] |= {key: value}

    count = {}
    file_name = path.split('/')[-1]
    print(file_name)
    # 统计包含 "True" 的键的数量
    for key in class_dict.keys():
        count[key] = sum(any(value == "True" for value in values) for values in class_dict[key].values())
        pre = count[key]/len(class_dict[key])
        
        # print(file_name, key, count[key])
        # print(file_name, key, pre)
        print(round(pre*100, 1), end=' &')
    # print(file_name, "total" ,sum(count.values())/len(data))
    print(round(sum(count.values())/len(data)*100, 1))
    print("-"*80)

