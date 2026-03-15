"""
扩展插件基类（预留接口）
自定义插件继承 BasePlugin 并实现 on_result 方法
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePlugin(ABC):
    """情感分析扩展插件基类"""

    @abstractmethod
    def on_result(self, result: Dict[str, Any]) -> None:
        """
        当有新的情感分析结果时触发
        :param result: 融合情感结果字典
        """
        pass


class ConsoleLogPlugin(BasePlugin):
    """示例插件：将结果额外输出到控制台"""

    def on_result(self, result: Dict[str, Any]) -> None:
        print(f"[Plugin] 情感: {result['emotion']} | 置信度: {result['confidence']:.0%}")


class DatabasePlugin(BasePlugin):
    """预留：数据库存储插件（未实现）"""

    def on_result(self, result: Dict[str, Any]) -> None:
        # TODO: 接入数据库存储情感日志
        pass


class WebSocketPlugin(BasePlugin):
    """预留：WebSocket推送插件（未实现）"""

    def on_result(self, result: Dict[str, Any]) -> None:
        # TODO: 通过WebSocket实时推送情感数据
        pass
