#导入 Python 系统模块，主要用于安全地读取环境变量（如 API Key）
import os

#导入 OpenAI 兼容的客户端。ChatOpenAI 用于大模型对话生成，OpenAIEmbeddings 用于将文本转换为向量（Embedding）
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 对话 Prompt 模板，用于规范化输入给大模型的结构化提示词
from langchain_core.prompts import ChatPromptTemplate

# LCEL 的一个核心工具，作用是原封不动地传递输入数据（后面会详细解释它在 RAG 中的关键作用）
from langchain_core.runnables import RunnablePassthrough

# 输出解析器，用于剥离大模型返回的复杂对象（AIMessage），只提取纯文本内容
from langchain_core.output_parsers import StrOutputParser

# LangChain 的核心数据类，用来包装一段文本内容（page_content）
from langchain_core.documents import Document

# 文本切分器。用于把长文档切成适合大模型处理的的小片段（Chunk）
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 导入向量数据库 Chroma。它负责存储文本向量，并支持高效的相似度检索（支持内存或本地持久化
# langchain_community包已经弃用，改用langchain_chroma引入Chroma
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

#用几段文本模拟“知识库”
docs = [
    Document(page_content="wlz公司创建于2020年9月1日，地点是在北京朝阳区双桥"),
    Document(page_content="注册资金10万元"),
    Document(page_content="公司员工10人"),
    Document(page_content="主要从事AI agent开发")
]

# 实例化一个递归字符文本切分器
# 每个文本块的最大字符数限制为 256
# 相邻两个块之间重叠 20 个字符，防止切断上下文的语义连贯性
splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap = 20)
# 执行切分
splits = splitter.split_documents(docs)

# 将切分后的文本（splits）交由 OpenAIEmbeddings 转成向量，并存入 Chroma 向量数据库中
# embedding 传入了 GLM（或兼容的 OpenAI 格式）的 Embedding 服务配置，用于计算文本的语义相似度
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(
        model="embedding-2",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key=os.getenv("ZHIPU_API_KEY")
    )
)
# vectorstore (向量数据库)：就像是一个装满无数资料页的大仓库。它里面存好了文字的数学向量，但它自己不会主动去找资料，它只是个静态的存储介质
# retriever (检索器)：相当于你在仓库门口雇了一个专职检索员。你给他一个问题，他的工作就是跑进仓库里翻找，并把最符合要求的资料挑出来交给你
# 把一个静态的向量数据库，包装成一个随时待命的“检索员（Retriever）”
# 每次检索时，最多只给我找出并返回最相关的 2 个文本片段
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 大模型初始化
llm = ChatOpenAI(
    model="deepseek-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Prompt 初始化
# 防止大模型凭空胡编乱造，即减少幻觉
prompts = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识库助手，请根据文档回答问题，文档中没有就说不知道，不要自己编造信息。"),
    ("human", "参考资料：{context}\n用户问题：{question}"),
])

# RAG 核心管道组装 (LCEL Chain)
# 用户调用invoke的问题，会同时给到 retriever 和 RunnablePassthrough()
# 把用户的问题交给前面定义的“检索员”（retriever）, 检索员拿着问题去向量数据库里搜，找出最相关的 2 条文本，拼成一个字符串输出给 context
# RunnablePassthrough 原封不动地把问题传给question
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompts # 上一步的字典会被喂给 prompt 模板。模板把 context 和 question 填入对应的占位符中，生成格式化的 ChatPromptValue（消息列表）
    | llm # 将格式化后的消息发送给 DeepSeek 大模型，大模型推理并返回一个 AIMessage 对象
    | StrOutputParser() # 从 AIMessage 中提取出纯文本内容，输出最终的字符串答案
)

response = rag_chain.invoke("介绍下wlz公司")
print(response)