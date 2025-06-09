import numpy as np
import pandas as pd
from .scenario_gen import ScenarioGen


class PastGen(ScenarioGen):
    """
    Implements azure blob file storage
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 
    def gen_scenarios(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Return the data with the given starting_index and ending_index
        """
        if self.use_log_returns:
            returns = np.log(returns + 1)
        return returns
    
