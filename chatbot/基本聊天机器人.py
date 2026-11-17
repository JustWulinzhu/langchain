import os
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
load_dotenv()

# LangChain 统一的三种核心对话角色消息类。大模型本身是无状态的，它靠这三种标签来区分每句话是谁说的
# SystemMessage：系统提示词（给 AI 设定人设和规则）
# HumanMessage：人类用户的提问
# AIMessage AI的回答
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

# 创建deepseek大模型
llm = ChatDeepSeek(
    model = os.getenv("DEEPSEEK_MODEL"),
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    temperature = 0 # 控制生成结果的随机性，通常适合希望回答相对稳定的场景
)

# 定义ai角色和职责
# 给ai的提示词。无论后面聊了多少轮，这条指令通常都会放在每次请求列表的最前面，用来约束 AI 的身份和回复语言。
system_message = SystemMessage(
    content = "你是一位专业的python助手，请用中文回答"
)

# 创建聊天历史存储变量
# 初始化一个空列表，充当机器人的“短期记忆库”。程序运行期间，所有的用户提问和 AI 回答都会按时间顺序存进这个列表里
chat_history = []

# 开始循环对话
while True:
    # 获取用户输入
    # 程序在此处暂停，在终端打印换行符 \n 和 用户： 提示符，等待你敲击键盘并按回车键，随后把你输入的字符串赋值给变量 user_input
    user_input = input("\n用户：")
    # 退出机制，判断是否需要终止循环退出
    if user_input.lower() in ["exit", "quit"]:
        print("再见")
        break
    # 判断是否清空聊天记录
    # 重置记忆机制（类似 ChatGPT 的“新建对话”）
    if user_input == "/clear":
        chat_history.clear()
        print("AI：聊天记录已清空")
        continue
    # 组装本轮请求消息
    # 作用：构造本轮要发给大模型的完整对话剧本
    # 原理：大模型本身没有记忆，所谓“多轮对话”，本质上就是每次发新问题时，把从第一轮到现在的所有聊天记录打包在一起，重新发给大模型看一遍。
    messages = [
        # 给ai的提示词
        system_message,
        # 假设之前已经聊过 1 轮，chat_history 里有 [HumanMessage("你好"), AIMessage("你好！")],加上 * 号后，Python 会把列表“拆开”，平铺放进外层的 messages 列表中
        *chat_history,
        # 刚刚输入的新问题
        HumanMessage(content = user_input),
    ]

    try:
        # 调用大模型
        response = llm.invoke(messages)
        answer = response.content
        print(f"\nAI: {answer}")
        # 保存本轮用户提问消息
        chat_history.append(HumanMessage(content = user_input))
        # 保存本轮AI回答的消息
        chat_history.append(AIMessage(content = answer))
    except Exception as e:
        print(f"系统异常：{e}")

# response.content：AI 返回的正文。
# response.response_metadata：响应相关元数据。
# response.usage_metadata：如果模型提供使用量信息，可以查看 Token 使用情况。
# print(response.content)