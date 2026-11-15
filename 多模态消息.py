import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

#多模态消息（Multimodal Messages）指的是：
# 消息的 content（内容）不再局限于单一的纯文本字符串，而是可以同时包含文本、图片、音频、视频甚至文件等多种不同媒体格式的混合结构。

llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL"),
)

# 构建多模态消息
# HumanMessage：代表来自人类用户的一条输入消息。
# 多媒体混合结构 (content 为列表)：
# 不同于传统的纯文本字符串，多模态消息的 content 是一个由多个字典组成的列表（List）：
# 文本块 (type: "text")：告诉大模型具体的指令——“分析下这张图片的内容”。
# 图片块 (type: "image_url")：严格遵循 OpenAI 多模态规范，通过 image_url 嵌套字典传入图片的公网访问地址。
message = HumanMessage(
    content = [
        {"type": "text", "text": "分析下这张图片的内容"},
        {
            "type": "image_url",
            "image_url": {"url": "https://pic.rmb.bdstatic.com/bjh/beautify/7edee34278ef5129a3e3233b9737807a.jpeg@h_1280"}
        }
    ]
)

agent = create_agent(
    model = llm,
    system_prompt = "你是一个专业的图片分析助手"
)

# 不再使用一次性阻塞的 invoke，而是用 stream（同步流式传输）
# stream_mode="messages" 参数告诉底层引擎：“当大模型每吐出一点文本片段（Token）时，就立刻通过生成器把消息推出来，不要等全部写完
result = agent.stream(
    {"messages": message},
    stream_mode = "messages"
)

# 遍历这个生成器。每次循环拿到的 token 就是大模型刚刚生成的微小文本片段。
for token, metadata in result:
    if token.content:
        # Python 默认的 print 打印完会自动换行，这里把结尾设为空字符串（不换行），从而在终端拼凑出流畅的“打字机实时效果”。
        print(token.content, end = "")