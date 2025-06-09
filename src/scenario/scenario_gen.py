"""
Contains the definition of abstract class ScenarioGen
"""
from abc import ABC, abstractmethod
import pandas as pd

class ScenarioGen(ABC):
    """
    Contains abstract methods for scenario generation
    """
    @abstractmethod
    def __init__(self, use_log_returns:bool = False, sample_size:int = 0):
        """
        Initialize the data source
        """
        self.use_log_returns = use_log_returns
        self.sample_size = sample_size

    @abstractmethod
    def gen_scenarios(self, returns:pd.DataFrame) -> pd.DataFrame:
        """
        Generate Scenarios for given number of days
        """

    