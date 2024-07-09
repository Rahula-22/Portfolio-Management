#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

def moment(series, degree):
    mean = np.mean(series)
    moment = np.mean((series - mean) ** degree)
    return moment

def skewness(returns):
    skew = moment(returns, 3) / (moment(returns, 2) ** 1.5)
    return skew

def kurtosis(returns):
    kurt = moment(returns, 4) / (moment(returns, 2) ** 2) - 3
    return kurt

def annualized_returns(returns):
    annual_return_rate = np.prod(1 + returns) ** (12 / len(returns)) - 1
    return annual_return_rate

def annualized_volatility(returns):
    stdv = np.sqrt(moment(returns, 2))
    volatility = stdv * np.sqrt(12)
    return volatility

def sharpe_ratio(returns, risk_free_rate=0.03):
    mean_portfolio_returns = np.mean(returns)
    standard_deviation = np.sqrt(moment(returns, 2))
    sharpe = (mean_portfolio_returns - risk_free_rate) / standard_deviation * np.sqrt(len(returns))
    return sharpe

def compound(r):
    return np.expm1(np.log1p(r).sum())

def jarque_bera_test(returns):
    statistic, p_value = st.jarque_bera(returns)
    return p_value > 0.05

def drawdown(return_series):
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns

def semi_deviation(returns):
    mean = np.mean(returns)
    semideviation = np.sqrt(np.mean(np.square(returns[returns < mean] - mean)))
    return semideviation

def historical_VaR(returns, confidence):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence) * len(sorted_returns))
    var = sorted_returns[index]
    return -var

def historical_CVaR(returns, confidence):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence) * len(sorted_returns))
    cvar = np.mean(sorted_returns[:index])
    return -cvar

def gaussian_VaR(returns, confidence):
    mean_return = np.mean(returns)
    stdv_return = np.sqrt(moment(returns, 2))
    z_value = st.norm.ppf(1 - confidence)
    var = mean_return + z_value * stdv_return
    return -var

def gaussian_CVaR(returns, confidence):
    var = gaussian_VaR(returns, confidence)
    cvar = np.mean([x for x in returns if x < var])
    return -cvar

def cornish_fisher_VaR(returns, confidence):
    mean = np.mean(returns)
    stdv = np.sqrt(moment(returns, 2))
    z = st.norm.ppf(1 - confidence)
    s = skewness(returns)
    k = kurtosis(returns)
    z_adj = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k - 3) / 24 - (2*z**3 - 5*z) * (s**2) / 36
    var = mean + z_adj * stdv
    return -var

def portfolio_return(weights, returns):
    #calculates portfolio returns given returns of the assets and their weights
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    #calculates portfolio volatility given covariance of the assets and their weights
    return (weights.T @ covmat @ weights)**0.5


def minimize_vol(target_return, er, cov):
    #outputs weights for assets to minimize volatility for a particular return rate of the portfolio
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n 

    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def msr(riskfree_rate, er, cov):
    #outputs weights for a portfolio having maximum sharpe ratio out of all possible portfolios
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n 
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def gmv(cov):
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def optimal_weights(n_points, er, cov):
    #gives optimal weights for a given return rate
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    #Plots the efficient frontier for a given portfolio of assets
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
        return ax



def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    
    #Runs a backtest of the CPPI strategy, given a set of returns for the risky asset
    # Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History

    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 

    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return backtest_result


def summary_stats(r, riskfree_rate=0.03):
    #Outputs a dataframe with basic analysis for a return series or dataframe
    ann_r = annualized_return(r)
    ann_vol = annualize_vol(r)
    ex_ret = [i - 0.03 for i in ann_r]
    ann_sr = ex_ret/ann_vol
    skew = skewness(r)
    kurt = kurtosis(r)
    cf_var5 = var_gaussian(r, modified = True)

    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Sharpe Ratio": ann_sr,
        
    }).reset_index()


# In[ ]:




