from mcp.server import MCPServer

# 创建一个 MCP Server，名字叫 weather-server。
# 可以理解为创建一个对外提供工具的服务。
mcp_server = MCPServer("weather-server")

# 定义一个查询天气的 mcp工具
# 它的作用是把下面的 Python 函数注册为 MCP Tool。原来普通函数只能被 Python 代码直接调用，现在它可以被 MCP Client 发现和调用。
# @mcp_server 怎么写 取决于前面 的MCPServer 变量名是怎么定义的，这里定义的是mcp_
@mcp_server.tool()
def get_weather(city: str) -> str:
    """"
        查询指定城市的天气
        参数：city 例如：北京 太原 上海等等
    """
    # 模拟天气api返回的数据
    weather_data = {
        "北京": {
            "天气": "多云",
            "温度": "15"
        },
        "太原": {
            "天气": "大暴雨",
            "温度": "11"
        },
        "上海": {
            "天气": "晴天",
            "温度": 30
        }
    }
    # 获取对应城市的数据
    result = weather_data.get(city)

    # 增加判断 处理不存在的城市
    if result is None:
        return f"没有{city}城市的天气信息"

    # 返回查询结果
    return (
        f"{city}城市天气：{result['天气']}，"
        f"温度：{result['温度']}"
    )

if __name__ == "__main__":
    mcp_server.run()

