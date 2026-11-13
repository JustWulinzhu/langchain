# 加载电脑里面的环境变量
import os
# Python 的异步 I/O 框架，用来处理“异步”任务的工具（可以让程序在等待大模型或网络回复时，不会卡死）
import asyncio
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
# 用来连接外部工具服务器（MCP 协议）的适配器，用于将大语言模型与外部数据源和工具服务进行解耦
from langchain.mcp import MCPAdapter
# 把.env里面的配置加载到环境变量里面
from dotenv import load_dotenv
load_dotenv()

# 初始化大模型
llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL")
)

# async def 定义一个“异步函数”，所有的动作都装到main里面
async def main() :
    # 创建MCP adapter，连接外部的天气小助手
    # with的作用：在 Python 中用来管理“资源”（比如打开文件），不管发生什么意外或者异常，都会自动释放资源，防止文件忘关或者内存溢出
    # async with 可以理解才异步的with，就是开启和关闭资源这个操作也是异步的
    # 进入代码块时：async with 会在后台异步地启动 python mcp/weather_server.py 这个子进程，并通过 stdio（标准输入输出）建立通信管道，然后把这个工具适配器命名为 adapter 供你使用
    # as adapter 给适配器起名，为啥叫适配器？因为LangChain是听不懂 MCP 服务器吐出来的数据，这里需要翻译成 LangChain看得懂的 Python 工具。
    async with MCPAdapter({
        "weather": {
            "command": "python",
            "args": [
                "mcp/weather_server.py",
            ],
            "transport": "stdio"
        }
    }) as adapter:
        # 获取mcp server提供的工具
        # await 等待天气服务器的返回，等待他给出工具清单
        # await 的意思就是清单没回来之前在这里等着，别着急执行下面的代码
        tools = await adapter.list_tools()

        # 创建langchain agent
        # 这里可以理解创建好了一个全能的 AI 秘书agent
        agent = create_agent(
            model = llm,
            tools = tools,
            system_prompt = "你是一个天气助手，用户询问天气时，必须调用工具查询天气，如果没有对应的天气数据要返回不知道，禁止编造数据"
        )

        # 调用agent ainvoke 异步任务，向AI 秘书agent 下达任务
        # 在 Python 中，同步世界和异步世界是泾渭分明、互不相通的两个宇宙：
        # 1、外层是异步宇宙：因为你的整个 main() 函数是用 async def 定义的，这就决定了你身处“异步宇宙”中。
        # 2、 规矩：在异步宇宙里，所有的耗时操作（无论是查本地子进程，还是连云端网络）都必须用异步的方式去等待（也就是用 await）。
        # 你不能在异步宇宙里突然强行插入一个同步的、死等的方法，否则整个事件循环的节奏就会被彻底打乱报错。
        ### 只要是涉及“把请求发出去、然后要花时间等对方给结果”的动作（无论对方是本地程序还是远端服务器），在异步函数里都必须用 await 去乖乖站好、等待收信！
        result = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": "今天天津天气怎么样？"
            }]
        })
        print(result["messages"][-1].content)

# 如果你是直接在终端敲命令运行这个文件（比如 python weather_agent.py），Python 就会把这个文件的名字自动设为 "__main__"（意思是：“你就是今天的主角/主程序”）。
# 检查一下：如果我是被直接运行的（作为主程序），那么就执行下面缩进里的代码；如果我只是被别人悄悄引用的，那就别瞎执行。
# 这是一个非常经典的 Python 编程规范，用来防止代码在被别人引用时发生意外的自动触发。
if __name__ == "__main__":
    # 启动异步引擎，把 main() 函数的所有流程跑起来
    asyncio.run(main())