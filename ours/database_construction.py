import wikipedia
import os
import json
from ragatouille.utils import get_wikipedia_page
from tqdm import tqdm


# 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:9098"
os.environ["https_proxy"] = "http://127.0.0.1:9098"


# 获取数据集的关键词
def get_keyword(dataset_path):
    # 存储最底层文件夹的名称
    bottom_level_dir_names = []
    # 遍历给定路径下的所有文件和文件夹
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        # 如果一个文件夹下没有其他文件夹，则认为它是最底层的
        if not dirnames:
            # 提取文件夹名称
            dir_name = os.path.basename(dirpath)
            bottom_level_dir_names.append(dir_name)
    return bottom_level_dir_names


# 查询最匹配关键词的前三个维基百科词条
def get_wikipedia_page_title(keyword):
    try:
        # 使用wikipedia的search方法搜索关键词，返回最匹配的10个结果
        page = wikipedia.search(keyword)[:10]
        return page
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    image_path = '../dataset'
    database = {}
    # 获取所有关键词
    keyword_list = get_keyword(image_path)
    # 获取每个关键词的词条
    for keyword in tqdm(keyword_list):
        # 获取每个词条的知识
        pages = get_wikipedia_page_title(keyword)
        database[keyword] = {}
        for page in pages:
            database[keyword][page] = get_wikipedia_page(page)
        # 保存到本地
        with open('../database/' + keyword + '.json', 'w') as json_file:
            json.dump(database[keyword], json_file, indent=4)
