
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import sys
import httpx
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


# 1. 加载数据
loader = TextLoader("../database/Black Cat Sheriff.json")
pages = loader.load()
# 2. 知识切片 将文档分割成均匀的块。每个块是一段原始文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
)
docs = text_splitter.split_documents(pages)
# 3. 利用embedding模型对每个文本片段进行向量化，并储存到向量数据库中
embed_model = OpenAIEmbeddings(
    openai_api_base="https://oneapi.xty.app/v1", 
    openai_api_key="sk-icjltWAeAZCp0oMNAcD970B5F88546169515B8995e66C389",
    client=httpx.Client(
        base_url="https://oneapi.xty.app/v1",
        follow_redirects=True,
    )
)
vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model , collection_name="openai_embed")
# 4. 通过向量相似度检索和问题最相关的K个文档。
query = "Black Cat Sheriff"
result = vectorstore.similarity_search(query ,k = 5)
for r in result:
    print(r)
sys.exit(0)
# 5. 原始query与检索得到的文本组合起来输入到语言模型，得到最终的回答
def augment_prompt(query: str):
    # 获取top3的文本片段
    results = vectorstore.similarity_search(query, k=2)
    source_knowledge = "\n".join([x.page_content for x in results])
    # 构建prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    contexts:
    {source_knowledge}

    query: {query}"""
    return augmented_prompt
print(augment_prompt(query))

# 创建prompt
prompt = HumanMessage(
    content=augment_prompt(query)
)
messages = [
    SystemMessage(content="You are a helpful assistant."),
]
messages.append(prompt)
chat = ChatOpenAI(
    openai_api_base="https://oneapi.xty.app/v1", 
    openai_api_key="sk-icjltWAeAZCp0oMNAcD970B5F88546169515B8995e66C389",
    model='gpt-3.5-turbo',
    client=httpx.Client(
        base_url="https://oneapi.xty.app/v1",
        follow_redirects=True,
    )
)
res = chat(messages)

print(res.content)