from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseEngine(ABC):
    @abstractmethod
    def load(self, file_path: str):
        pass
    
    @abstractmethod
    def predict(self, input_data: dict) -> dict:
        pass

    @abstractmethod
    async def stream(self, input_data: dict) -> AsyncGenerator[dict, None]:
        pass