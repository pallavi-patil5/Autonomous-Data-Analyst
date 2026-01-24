"""
Base Agent Protocol - Robust, Safe, Self-Describing
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, get_type_hints
import inspect


class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    Adds:
    ✅ Safe execution (never crashes)
    ✅ Auto-parameter filtering
    ✅ Better error messages
    ✅ Optional dataset checking
    """

    def __init__(self):
        self.agent_name = self.__class__.__name__

    @abstractmethod
    def get_capabilities(self) -> List[Dict[str, Any]]:
        pass

    def get_all_methods(self) -> List[str]:
        return [
            method for method in dir(self)
            if callable(getattr(self, method))
            and not method.startswith('_')
            and method not in ['get_capabilities', 'get_all_methods', 'describe', 'execute_capability']
        ]

    def describe(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "capabilities": self.get_capabilities(),
            "available_methods": self.get_all_methods()
        }

    def _filter_valid_params(self, method, kwargs):
        """
        Removes parameters not accepted by the method.
        Prevents unexpected parameter errors.
        """
        sig = inspect.signature(method)
        valid_params = {}

        for name, value in kwargs.items():
            if name in sig.parameters:
                valid_params[name] = value

        return valid_params

    def execute_capability(self, function_name: str, **kwargs) -> Any:
        """
        Safe dynamic execution.
        ✅ No hard crashes
        ✅ Filters invalid params
        ✅ Checks dataset availability
        """

        if not hasattr(self, function_name):
            return f"⚠️ Method '{function_name}' not found in {self.agent_name}."

        method = getattr(self, function_name)

        # Dataset safety check
        if "data" in self.__dict__:
            if self.__dict__["data"] is None:
                return f"⚠️ No dataset loaded in {self.agent_name}. Please load data first."

        # Filter parameters
        safe_kwargs = self._filter_valid_params(method, kwargs)

        try:
            return method(**safe_kwargs)
        except Exception as e:
            return f"❌ Error executing {function_name} in {self.agent_name}: {str(e)}"
