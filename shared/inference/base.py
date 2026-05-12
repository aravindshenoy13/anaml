from abc import ABC, abstractmethod


class BaseEngine(ABC):
    @abstractmethod
    def load(self, weight_path: str):
        pass
    
    @abstractmethod
    def predict(self, input_data: dict) -> dict:
        pass

    @abstractmethod
    async def stream(self, input_data: dict):
        pass

    @abstractmethod
    def metadata(self):
        pass