#logt

import statistics
import numpy as np
import pandas as pd
import yfinance as yf
import sys
from scipy import stats

uppers = [arg.upper() for arg in sys.argv]
if 'NOPLOT' not in uppers:
	import matplotlib.pyplot as plt

ticker = input("ticker: ").upper()
try:
	data = pd.read_pickle(ticker+'.pkl')
	print("Previous data found")
except:
	data = yf.download(ticker, period='max')
	if data.empty:
		exit()
	data.to_pickle(ticker+'.pkl')

data = data.reset_index()
data.loc[0, 'percent_change'] = 0
for i in range(1, len(data)):
	data.loc[i, 'percent_change'] = 100* (data.loc[i, 'Close'] - data.loc[i-1, 'Close']) / data.loc[i-1, 'Close']

nu, mu, tau = stats.t.fit(data['percent_change'])
print("T fit: DF={:.6f}, LOC={:.6f}, SCALE={:.6f}".format(nu, mu, tau))

if 'NOPLOT' not in uppers:
	plt.ion()
	plt.show()

prices = [data.loc[len(data) - 1, 'Close']] * 252

'''
SPY
nu=2.790560
mu=0.071185
tau=0.702406
'''

while True:
	percent_change = mu + tau*np.random.standard_t(nu)
	prices.append(prices[-1] * (1+percent_change/100))
	print("${:.2f}".format(prices[-1]))
	prices.pop(0)
	if 'NOPLOT' not in uppers:
		plt.clf()
		plt.plot(prices)
		plt.pause(0.1)