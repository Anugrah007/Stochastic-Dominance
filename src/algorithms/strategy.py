
import pandas as pd
import numpy as np
from scipy import optimize
from abc import ABC, abstractmethod
import cvxpy as cp
from src.datasource.yahoodata import YahooDataSource
from src.scenario.scenario_gen import ScenarioGen


class Strategy(ABC):
    """
    Contains abstract methods for scenario generation
    """

    @abstractmethod
    def get_optimal_allocations(self,*args,**kwargs):
        """
        Get the Optimal weights 
        """

    @abstractmethod
    def run_strategy(self,*args,**kwargs):
        """
        Run strategy between this ind
        """

class ConstrainedBasedStrategy(Strategy):
    

    def run_strategy(self, data_source:YahooDataSource, test_steps: int = 12, rebalancing_frequency_step: int = 1, start_date: str = None, end_date: str = None, data_frequency: str = '1MS', scenario_generator: ScenarioGen = None):

        weights_dict = {}

        start_date = start_date if start_date else data_source.start_date
        end_date = end_date if end_date else data_source.end_date
        
        date_range = pd.date_range(start_date, end_date, freq=data_frequency)

        for index, date in enumerate(date_range[test_steps::rebalancing_frequency_step]):

            test_start_date = date_range[index*rebalancing_frequency_step]
            test_end_date = date

            price_data = data_source.get_data_by_frequency(start_date = test_start_date, end_date = test_end_date, frequency = data_frequency)

            
            rtn_data = price_data.pct_change()[1:]
            
            # generate scenarios using vine copula
            if scenario_generator:
                scenarios = scenario_generator.gen_scenarios(rtn_data)

            wealth_allocations = self.get_optimal_allocations(scenarios.T if scenario_generator else rtn_data.T,1)
            
            weights_dict[date] = dict(zip(price_data.columns,wealth_allocations))

        weights_dict = pd.DataFrame(weights_dict).T
        weights_dict.index = pd.to_datetime(weights_dict.index)
        
 
        return weights_dict


class CvarMretOpt(ConstrainedBasedStrategy):


    def __init__(self,ratio=0.5,risk_level=0.1):


        self.ratio = ratio
        self.risk_level = risk_level
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None
        

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):
        self.array = returns_data.to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount 
        self.results = self.optimize(self.ratio,self.risk_level)
        return self.results.x[self.num_senarios:self.num_senarios+self.num_assets]
    
    def get_cvar_value(self):
        return self.results.x[-1]

    def optimize(self,ratio,risk_level):

        """Solve the problem of minimizing the function 
                -(1-c) E[Z(x)] + c AVaR[Z(x)]
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))

        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets+1))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) # Rk  
            lhs_ineq[i,-1] = 1    # n

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets+1))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_senarios):
            bnd.append((0,float('inf')))

        for i in range(self.num_assets+1):
            bnd.append((0,float('inf')))
            
        bnd[-1] = (float('-inf'),float('inf'))


        obj = np.ones((1,self.num_senarios+self.num_assets+1))*(1/risk_level)*(1/self.num_senarios)*(ratio)
        obj[0,-1] = -1*(ratio)
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*(1-ratio)*np.array(np.transpose(mean))
        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")

        return opt

        

class MeanSemidevOpt(ConstrainedBasedStrategy):

    def __init__(self,ratio):
       
        self.ratio = ratio 
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None
        
    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount
        self.results = self.optimize(self.ratio)
        return self.results.x[self.num_senarios:self.num_senarios+self.num_assets]

    def optimize(self,ratio):

        """
            ρ =  Hybrid Risk Measure
            ρ = [Z(x)] = -E[Z(x)] + c σ[Z(x)]

            where σ[Z] = E{ max(0,E[Z] – Z)} is the lower semideviation of the first order.
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))
        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) + np.transpose(mean)  # Rk

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_senarios+self.num_assets):
            bnd.append((0,float('inf')))

        obj = np.ones((1,self.num_senarios+self.num_assets))*(1/self.num_senarios)*ratio
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*np.array(np.transpose(mean))

        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")
        return opt
    

class EqualyWeighted(ConstrainedBasedStrategy):

    def __init__(self):
    
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.to_numpy()
        self.num_assets = len(self.array[:,0])
        return (np.ones(self.num_assets)/self.num_assets)*investment_amount
    


class MeanSemiEquallyWeighted(ConstrainedBasedStrategy):

    def __init__(self,ratio):
       
        self.ratio = ratio 
        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.results = None
        
    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = 1
        self.results = self.optimize(self.ratio)
        allocations = np.array(self.results.x[self.num_senarios:self.num_senarios+self.num_assets])
        return (allocations + (np.ones(self.num_assets)/self.num_assets))/(np.sum(allocations) + (np.sum((np.ones(self.num_assets)/self.num_assets))))

    def optimize(self,ratio):

        """
            ρ =  Hybrid Risk Measure
            ρ = [Z(x)] = -E[Z(x)] + c σ[Z(x)]

            where σ[Z] = E{ max(0,E[Z] – Z)} is the lower semideviation of the first order.
        """

        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))
        lhs_ineq = np.zeros((self.num_senarios,self.num_senarios+self.num_assets))

        for i in range(self.num_senarios):
            lhs_ineq[i,i] = -1  # vk
            lhs_ineq[i,self.num_senarios:self.num_senarios+self.num_assets] = -1*(self.array[:,i]) + np.transpose(mean)  # Rk

        rhs_ineq = np.zeros((1,self.num_senarios))

        lhs_eq = np.zeros((1,self.num_senarios+self.num_assets))
        lhs_eq[0,self.num_senarios:self.num_senarios+self.num_assets] = 1
        rhs_eq = [self.investment_amount]

        bnd = []
        for i in range(self.num_senarios+self.num_assets):
            bnd.append((0,float('inf')))

        obj = np.ones((1,self.num_senarios+self.num_assets))*(1/self.num_senarios)*ratio
        obj[0,self.num_senarios:self.num_senarios+self.num_assets] = -1*np.array(np.transpose(mean))

        opt = optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,A_eq=lhs_eq,b_eq=rhs_eq, bounds=bnd,method="highs")
        return opt

    

class MeanVariance(ConstrainedBasedStrategy):

    
    def __init__(self,target_return=None,allow_shorting=False):

        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.target_return =  target_return
        self.allow_shorting = allow_shorting
        
    def calculate_mean(self) -> pd.Series:
        mean_returns = self.returns.mean()
        return mean_returns

    def calculate_covariance_matrix(self) -> pd.DataFrame:
        covariance_matrix = self.returns.cov()
        return covariance_matrix

    def portfolio_variance(self, weights):
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def min_variance(self, allow_shorting=False):
        num_assets = len(self.mean_returns)
        initial_weights = np.ones(num_assets) / num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if allow_shorting:
            bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))  
        else:
            bounds = tuple((0, float('inf')) for _ in range(num_assets))  
        
        result = optimize.minimize(self.portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result
    
    def min_variance_allocation(self):
        
        result = self.min_variance()
        if result.success:
            return dict(zip(self.returns.columns, result.x))
        else:
            raise ValueError("Optimization failed: " + result.message)

    def get_max_return(self) -> float:
        return self.mean_returns.max()

    def minimize_func(self, weights):
        return np.matmul(np.matmul(np.transpose(weights), self.cov_matrix), weights)
    
    def optimize(self, target_return=None, allow_shorting=False):
        
        num_assets = len(self.mean_returns)
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda weights: np.dot(weights, self.mean_returns) - target_return})
        
        if allow_shorting:
            bounds = tuple((float('-inf'), float('inf')) for _ in range(num_assets))
        else:
            bounds = tuple((0, float('inf')) for _ in range(num_assets))
        
        initial_weights = np.ones(num_assets) / num_assets
        result = optimize.minimize(self.minimize_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.returns =  returns_data.T
        self.mean_returns = self.calculate_mean().to_numpy()
        self.cov_matrix = self.calculate_covariance_matrix().to_numpy()

        result = self.optimize(target_return=self.target_return,allow_shorting=self.allow_shorting)
        if result.success:
            print("Optimal Weights:", result.x)
            return result.x
        else:
            raise ValueError("Optimization failed: " + result.message)
        



class StochasticDominance(ConstrainedBasedStrategy):

    def __init__(self,benchmark_srategy:ConstrainedBasedStrategy=EqualyWeighted(),long_only=True):
       

        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.strategy = benchmark_srategy
        self.long_only = long_only
        self.results = None
        
    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount
        bench_mark_weights = self.strategy.get_optimal_allocations(returns_data,investment_amount)
        print("Bench mark weights",bench_mark_weights)
        self.results = self.optimize(bench_mark_weights,long_only=self.long_only)
        return self.results

    def optimize(self,bench_mark_weights,long_only=True):


        chi = 0
        assets = self.num_assets
        senarios = self.num_senarios
        returns = self.array_transpose
        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))

        # Y_weights = (1/assets)*(np.ones((assets,1)))
        Y_returns = np.sort(((returns)@bench_mark_weights).flatten())
        V = []
        for eta in Y_returns:
            v_j = np.sum((eta-Y_returns)[Y_returns< eta])/(len(Y_returns))
            V.append(v_j)

        weights  = cp.Variable(shape=(assets,1),name="weights")
        S = cp.Variable(shape=(senarios,senarios),name="slack")
        X_returns = returns@weights

        constraints = []
        for j,eta in enumerate(Y_returns):
            for i,x in enumerate(X_returns):
                constraints.append(x+S[i,j]>=eta)

        for j,v_j in enumerate(V):
            constraints.append((1/(len(Y_returns)))*cp.sum(S[:,j])<=v_j)

        if long_only:
            constraints.append(weights>=0)
        constraints.extend([cp.sum(weights)==1,S>=0])

        # Diversification constraints
        # for i in range(assets):
        #     constraints.append(cp.abs(weights[i]) <= 1/assets)
        #     # constraints.append(cp.abs(weights[i]) >= 0.05)

        objective = cp.Maximize((mean.T@weights)- chi*cp.mean(cp.abs(X_returns-cp.mean(X_returns,axis=0))))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        try:
            return weights.value.flatten()
        except:
            return bench_mark_weights.flatten()

        return 
    

class StochasticDominanceOpt(ConstrainedBasedStrategy):

    def __init__(self,benchmark_srategy:ConstrainedBasedStrategy=EqualyWeighted(),long_only=True):
       

        self.results = None
        self.array = None
        self.num_assets = None
        self.num_senarios = None
        self.array_transpose = None
        self.investment_amount = None
        self.strategy = benchmark_srategy
        self.long_only = long_only
        self.results = None
        
    def get_optimal_allocations(self,returns_data:pd.DataFrame,investment_amount:int=1):

        self.array = returns_data.to_numpy()
        self.num_assets = len(self.array[:,0])
        self.num_senarios = len(self.array[0,:])
        self.array_transpose = np.transpose(self.array)
        self.investment_amount = investment_amount
        bench_mark_weights = self.strategy.get_optimal_allocations(returns_data,investment_amount)
        self.results = self.optimize(bench_mark_weights,long_only=self.long_only)
        return self.results

    def optimize(self,bench_mark_weights,long_only=True):


        chi = 0
        assets = self.num_assets
        senarios = self.num_senarios
        returns = self.array_transpose
        mean = np.resize(self.array.mean(axis=1),(self.num_assets,1))

        Y_weights = (1/assets)*(np.ones((assets,1)))
        Y_returns = np.sort(((returns)@Y_weights).flatten())
        V = [np.sum((eta-Y_returns)[Y_returns< eta])/(len(Y_returns)) for eta in Y_returns]

        dict_eta_V = dict(zip(Y_returns,V))


        k=0
        Eta = {Y_returns[-1]:Y_returns<=Y_returns[-1]}
        while True:

            weights = cp.Variable(shape=(assets,1),name="weights")
        
            objective = cp.Maximize((mean.T@weights))  # Objective function for first stage problem

        
            constraints = []
            for et in Eta:
                events = Eta[et]
                g_x_events = returns[events,:]@(weights)
                constraints.append(((1/(len(events)))*cp.sum(et -g_x_events )) <= dict_eta_V[et])
            
            constraints.extend([cp.sum(weights)==1,weights>=0])

            # Solve Problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            Z_x =returns@(weights.value).flatten()

            # Calculate deltas 
            delta_j = [ np.sum((eta-Z_x)[Z_x< eta])/(len(Z_x))-dict_eta_V[eta] for eta in Y_returns ]
            
            # Find out max eta
            delta_max = np.max(delta_j)
            eta_max = Y_returns[np.argmax(delta_j)]

            if delta_max <= 0:
                break
            else:
                Eta[eta_max] = Z_x<eta_max
                
            k= k+1

            if k>100:
                return bench_mark_weights.flatten()


        return weights.value.flatten()