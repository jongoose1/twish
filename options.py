import yfinance as yf
import datetime
import numpy as np
from scipy import optimize
from scipy.stats import norm

def d1(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility):
	return (np.log(stock_price/strike_price) + (risk_free_rate-dividend_yield+volatility**2/2)*years_to_expiry) / (volatility*np.sqrt(years_to_expiry))

def d2(d_1, years_to_expiry, volatility):
	return d_1 - volatility * np.sqrt(years_to_expiry)

def BSM_CALL(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility):
	d_1 = d1(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility)
	return stock_price * np.exp(-dividend_yield*years_to_expiry) * norm.cdf(d_1) - strike_price * np.exp(-risk_free_rate*years_to_expiry) * norm.cdf(d2(d_1, years_to_expiry, volatility))

def BSM_PUT(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility):
	d_1 = d1(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility)
	return strike_price * np.exp(-risk_free_rate*years_to_expiry) * norm.cdf(-d2(d_1, years_to_expiry, volatility)) - stock_price * np.exp(-dividend_yield*years_to_expiry) * norm.cdf(-d_1)

def call_iv_error(iv_guess, *args):
	stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, call_price = args
	return (BSM_CALL(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, iv_guess)-call_price)**2

def put_iv_error(iv_guess, *args):
	stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, put_price = args
	return (BSM_PUT(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, iv_guess)-put_price)**2

def ImpliedVolatilityCall(guess, *args):
	return optimize.fmin(func=call_iv_error, x0=guess, args=args, disp=False)[0]

def ImpliedVolatilityPut(guess, *args):
	return optimize.fmin(func=put_iv_error, x0=guess, args=args, disp=False)[0]

def ImpliedProbabilityCall(d_2):
	return norm.cdf(d_2)

def ImpliedProbabilityPut(d_2):
	return norm.cdf(-d_2)

def delta_call(d_1, years_to_expiry, dividend_yield):
	return np.exp(-dividend_yield*years_to_expiry) * norm.cdf(d_1)

def delta_put(d_1, years_to_expiry, dividend_yield):
	return np.exp(-dividend_yield*years_to_expiry) * (norm.cdf(d_1) - 1)

def gamma(d_1, stock_price, years_to_expiry, dividend_yield, volatility):
	return np.exp(-dividend_yield*years_to_expiry) * norm.pdf(d_1) / (stock_price * volatility * np.sqrt(years_to_expiry))

def theta_call(d_1, d_2, stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility):
	return (-(stock_price*volatility*np.exp(-dividend_yield*years_to_expiry)*norm.pdf(d_1)/(2*np.sqrt(years_to_expiry)))-risk_free_rate*strike_price*np.exp(-risk_free_rate*years_to_expiry)*norm.cdf(d_2)+dividend_yield*stock_price*np.exp(-dividend_yield*years_to_expiry)*norm.cdf(d_1)) / 365.25

def theta_put(d_1, d_2, stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility):
	return (-(stock_price*volatility*np.exp(-dividend_yield*years_to_expiry)*norm.pdf(d_1)/(2*np.sqrt(years_to_expiry)))-risk_free_rate*strike_price*np.exp(-risk_free_rate*years_to_expiry)*norm.cdf(-d_2)+dividend_yield*stock_price*np.exp(-dividend_yield*years_to_expiry)*norm.cdf(-d_1)) / 365.25

def vega(d_1, stock_price, years_to_expiry, dividend_yield, volatility):
	return stock_price*np.exp(-dividend_yield*years_to_expiry)*np.sqrt(years_to_expiry)*norm.pdf(d_1) / 100

def rho_call(d_2, strike_price, years_to_expiry, risk_free_rate):
	return strike_price * years_to_expiry * np.exp(-risk_free_rate*years_to_expiry)*norm.cdf(d_2) / 100

def rho_put(d_2, strike_price, years_to_expiry, risk_free_rate):
	return -strike_price * years_to_expiry * np.exp(-risk_free_rate*years_to_expiry)*norm.cdf(-d_2) / 100

def dte(option):
	# option str %Y-%m-%d
	return (
		datetime.datetime.strptime(option, "%Y-%m-%d")
		- datetime.datetime.combine(datetime.date.today(), datetime.time())
	).days

def get_rfr():
	try:
		risk_free_rate = yf.Ticker("^IRX").info['dayLow']/100
	except:
		print("Using fake risk free rate")
		risk_free_rate = 0.05
	return risk_free_rate

'''
Example option:

contractSymbol              SPY240719C00650000
lastTradeDate        2024-07-19 18:01:02+00:00
strike                                   650.0
lastPrice                                 0.01
bid                                        0.0
ask                                       0.01
change                                     0.0
percentChange                              0.0
volume                                   501.0
openInterest                              7326
impliedVolatility                     0.984375
inTheMoney                               False
contractSize                           REGULAR
currency                                   USD
'''
