import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Pydantic 本质上是 Python 的一个数据验证与解析库。结合你之前看过的 LangChain 结构化输出，
# 可以把它理解为：你用它来画一个“数据表格”，然后 Pydantic 负责严格检查输入的数据是否完全符合这个表格的格式。不符合就直接报错，绝不蒙混过关。
# 其实就是数据定义和验证的包，定义字段名称、类型、数据、字段说明等
from pydantic import BaseModel, Field

# 结构化输出，让ai按照该类定义的数据结构返回
# description是干啥的：这是 Pydantic 在 AI 场景下的独有价值：description 不是写给人看的注释，是直接喂给模型的提示语。所以描述写得越清楚，模型填得越准。
class OrderFormat(BaseModel) :
    is_found: bool = Field(description = "是否找到对应的订单") # 增加布尔类的字段，标记这次查询是否查到了，true/false
    order_id: str | None = Field(description = "订单id，没有查询到时为null")
    amount: str | None = Field(description = "订单金额，没有查询到时为null")
    status: str | None = Field(description = "订单状态，没有查询到时为null")
    logistics: str | None = Field(description = "物流状态，没有查询到时为null")
    result_message: str = Field(description = "查询的结果说明，是否找到了对应的订单")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

async def main() :
    config = {
        "mcpServers": { # MCPAdapter（底层基于的 FastMCP 客户端规范）只认驼峰命名的 "mcpServers"
            "order": {
                "command": "python",
                "args": [
                    "mcp/order_server.py"
                ],
                "transport": "stdio"
            }
        }
    }
    async with MCPAdapter(config) as adapter :
        tools = await adapter.list_tools()
        for tool in tools :
            print(tool.name)

        agent = create_agent(
            model = llm,
            tools = tools,
            response_format = OrderFormat,
            system_prompt = """
            你是一个订单查询小助手，
            用户查询订单信息、物流信息等需要调用工具查询，
            禁止编造数据
            """
        )

        result = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": "帮我查询订单order3，告诉我订单详情，金额、物流等"
                }
            ]
        })

        # 这两处用法符合 LangChain 当前 Agent 结构化输出的设计。
        # Agent 会把结构化结果放在最终状态的 structured_response 字段中。
        structure_data = result.get("structured_response")
        print(structure_data)
        if structure_data.is_found:
            print("订单id：" + structure_data.order_id + "，订单金额：" + structure_data.amount + "，订单状态：" + structure_data.status + "，物流状态：" + structure_data.logistics)
        else:
            print(structure_data.result_message)
if __name__ == "__main__":
    # 启动异步引擎，把 main() 函数的所有流程跑起来
    asyncio.run(main())