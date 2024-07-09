#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import minimize
import yfinance as yf
import seaborn as sns
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import Risk_kit as rk


# In[37]:


#getting data from yfinance
tickers = ['RVNL.NS', 'IRFC.NS', 'RECLTD.NS', 'NHPC.NS', 'POWERGRID.NS', 'ZOMATO.NS', 'COCHINSHIP.NS', 'IRB.NS', 'SJVN.NS', 'NBCC.NS']
data = yf.download(tickers, start='2021-7-9', end='2024-7-9') #fetching monthly data for 3 years
price_data=data['Close'] #taking the closing price to determine returns
returns = price_data[tickers].pct_change().dropna()
returns.dropna(inplace=True)
rets = pd.DataFrame(returns)


# In[38]:


cov_matrix = rets.cov() #to get covariance between the assets
cov_matrix.head()


# In[39]:


#calculating annualized returns using the function from the risk module
ann_rets = pd.Series(rk.annualized_returns(rets))
ann_rets


# In[40]:


#plotting the annual returns of the assets
column_names = price_data.columns
ann_rets.index=column_names
ann_rets.sort_values().plot.bar()


# In[47]:


#calculating weights to optimize for the maximum sharpe ratio using msr function from the risk module
weights=rk.msr(0.03,ann_rets, cov_matrix)
#return of the optimized portfolio
r=rk.portfolio_return(weights,ann_rets)
#volatility of the optimized portfolio
vol=rk.portfolio_vol(weights,cov_matrix)


# In[48]:


#PLotting the efficient frontier, and locating the portfolio having maximum Sharpe ratio
rk.plot_ef(50, ann_rets, cov_matrix)
#Plotting the line for returns of the risk free asset (risk free returns = 3%)
x=[0,0.020]
y=[0.03,0.020*(r-0.03)/vol+0.03]
plt.title("Efficient Frontier")
plt.plot(x,y,label='Max sharpe portfolio',color="green")
plt.plot(vol,r, color='green', marker='.',markersize=5, markeredgecolor= 'blue',label="Maximum Sharpe Portfolio")
plt.ylabel("Returns")
plt.legend()
plt.show()


# In[49]:


sharpe = (r-0.03)/vol
print(f"The return of the portfolio is {r*100}%")
print(f"The volatility of the portfolio is {vol*100}%")
print(f"The Sharpe Ratio of the portfolio is {sharpe}")


# In[ ]:


#CPPI


# In[50]:


#Taking the same 10 assets as risky assets and using them as risky assets in CPPI
df = rk.run_cppi( rets,safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None)
# plotting the wealth index after running cppi
ax = df["Wealth"].plot(figsize=(12,5))   


# In[51]:


#plotting returns if all the initial amount was placed in the risky asset
df["Risky Wealth"].plot()


# In[52]:


#calculating the return series for the portfolio where cppi was used
cppi_rets = df["Wealth"].pct_change().dropna()


# In[53]:


#performance of the portfolio
rk.summary_stats(cppi_rets)


# In[ ]:




