import os
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

directory_path="../results/image_identification"
all_file_paths = get_all_file_paths(directory_path)

for path in all_file_paths:
    with open(path, 'r') as file:
        data = json.load(file)


    class_dict = {'a': {}, 'b': {}, 'c': {}, 'd': {}, 'e': {}}

    for key, value in data.items():
        class_dict[key[0]] |= {key: value}

    total_true = []
    total_pre = []
    print(path.split("/")[-1])
    for type, type_data in class_dict.items():
        true_labels = []
        predicted_labels = []
    # 遍历JSON中的数据
        for key, predicted_value in type_data.items():
            # 提取真实值
            if 't' in key:
                true_value = 'yes'
            elif 'f' in key:
                true_value = 'no'
            else:
                continue  # 忽略无效的键
            
            
            if predicted_value.lower().strip().startswith("this image aligns with") or predicted_value.lower().strip().startswith("yes"):
                predicted_label = 'yes'
            # elif "not" in predicted_value.lower():
            #     predicted_label = 'no'  
            elif predicted_value.lower().strip().startswith("this image does not align with") or predicted_value.lower().strip().startswith("no"):
                predicted_label = 'no'
            # else:
            #     predicted_label = 'no'

            # 存储真实值和预测值
            total_true.append(true_value)
            total_pre.append(predicted_label)

            true_labels.append(true_value)
            predicted_labels.append(predicted_label)

        # 计算P, R, F1和准确率
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary', pos_label='yes')
        accuracy = accuracy_score(true_labels, predicted_labels)
        # print(len(predicted_labels))
        # print(f'{precision*100:.2f}\\%', end=' & ')
        # print(f'{recall*100:.2f}\\%',end=' & ')
        # print(f'{f1*100:.2f}\\%')
        print(f'{accuracy*100:.1f}', end=' &')

    precision, recall, f1, _ = precision_recall_fscore_support(total_true, total_pre, average='binary', pos_label='yes')
    accuracy = accuracy_score(total_true, total_pre)
    # print(len(total_pre))
    # print(f'{precision*100:.2f}\\%', end=' & ')
    # print(f'{recall*100:.2f}\\%',end=' & ')
    # print(f'{f1*100:.2f}\\%')
    print(f'{accuracy*100:.1f}')
    print()
