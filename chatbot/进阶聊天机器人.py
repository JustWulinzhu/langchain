import os
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
load_dotenv()

# LangChain 统一的三种核心对话角色消息类。大模型本身是无状态的，它靠这三种标签来区分每句话是谁说的
# SystemMessage：系统提示词（给 AI 设定人设和规则）
# HumanMessage：人类用户的提问
# AIMessage AI的回答
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中配置 DEEPSEEK_API_KEY")

model = os.getenv("DEEPSEEK_MODEL")
if not model:
    raise ValueError("请在 .env 文件中配置 DEEPSEEK_MODEL")

# 创建deepseek大模型
llm = ChatDeepSeek(
    model = model,
    api_key = api_key,
    temperature = 0, # 控制生成结果的随机性，通常适合希望回答相对稳定的场景
    timeout = 60,
    max_retries = 2
)

# 定义系统提示词
SYSTEM_PROMPT = """
你是一个专业的 AI 助手。

要求：
1. 使用中文回答用户。
2. 回答尽量清晰、准确。
3. 编程问题尽量提供代码示例。
4. 如果不知道答案，不要编造。
5. 对于复杂问题，分步骤解释。
"""

# 定义ai角色和职责
# 给ai的提示词。无论后面聊了多少轮，这条指令通常都会放在每次请求列表的最前面，用来约束 AI 的身份和回复语言。
system_message = SystemMessage(
    content = SYSTEM_PROMPT
)

# 创建聊天历史存储变量
# 初始化一个空列表，充当机器人的“短期记忆库”。程序运行期间，所有的用户提问和 AI 回答都会按时间顺序存进这个列表里
chat_history = []

# 定义聊天函数
def chat(user_input):
    global chat_history

    # 组装给AI的完整消息
    messages = [
        system_message,
        *chat_history,
        HumanMessage(content = user_input)
    ]

    try:
        # 调用大模型，流式接收回答
        print("\nAI: ", end = "", flush = True)
        answer = ""
        response = llm.stream(messages)
        for chunk in response:
            content = chunk.content
            print(content, end="", flush=True)
            answer += content
        print()

        # 保存用户消息
        chat_history.append(
            HumanMessage(content = user_input)
        )

        # 保存AI的回答
        chat_history.append(
            AIMessage(content = answer)
        )

        # 只保留最近 20 条消息
        chat_history = chat_history[-20:]
    except Exception as e:
        print(f"系统异常: {e}")

# 主程序入库
def main():
    # 默认提示
    print("=" * 50)
    print("欢迎使用 AI 对话机器人")
    print("模型：", model)
    print("输入 exit 退出")
    print("输入 /clear 清空聊天记录")
    print("=" * 50)

    while True:
        user_input = input("用户：")
        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("AI：再见！")
            break

        if user_input == "/clear":
            chat_history.clear()
            print("AI：聊天记录已清空。")
            continue

        chat(user_input)

if __name__ == "__main__":
    main()