import yfinance as yf
import logging
from pprint import pprint
import pandas_market_calendars as mcal
import datetime
import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import norm

pd.set_option('display.max_rows', None)

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

def delta_call(d_1, years_to_expiry, dividend_yield):
	return np.exp(-dividend_yield*years_to_expiry) * norm.cdf(d_1)

def delta_put(d_1, years_to_expiry, dividend_yield):
	return np.exp(-dividend_yield*years_to_expiry) * (norm.cdf(d_1) - 1)

def gamma(d_1, stock_price, years_to_expiry, dividend_yield, volatility):
	return np.exp(-dividend_yield*years_to_expiry) * norm.pdf(d_1) / (stock_price * volatility * np.sqrt(years_to_expiry))

def theta_call(d_1, stock_priceferf,,,):
	return (-(stock_price*volatility*np.exp(-dividend_yield*years_to_expiry)*norm.pdf(d_1)/(2*np.sqrt(years_to_expiry)))-risk_free_rate*strike_price*np.exp(-risk_free_rate*years_to_expiry)*norm.cdf(d_2)+dividend_yield*stock_price*np.exp(-dividend_yield*years_to_expiry)*norm.cdf(d_1)) / 365.25


cal = mcal.get_calendar("NYSE")
today = datetime.datetime.combine(datetime.date.today(), datetime.time())
today_str = today.strftime("%Y-%m-%d")
risk_free_rate = yf.Ticker("^IRX").info['dayLow']/100

'''
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

logging.getLogger('yfinance').disabled = True

while(True):
	ticker = input("TICKER: ").upper()
	if ticker == "exit":
		exit()
	quote = yf.Ticker(ticker)
	info = quote.info
	if 'symbol' not in info:
		print("invalid ticker")
		continue
	options = quote.options
	stock_price = (info['bid'] + info['ask']) / 2 
	if 'yield' in info:
		dividend_yield = info['yield']
	else:
		dividend_yield = 0
	for option in options:
		bdi = mcal.date_range(
			cal.schedule(
				start_date = today_str,
				end_date = option),
			frequency = '1D')
		bdi = bdi.normalize()
		s = pd.Series(data = 1, index=bdi)
		cdi = pd.date_range(
			start = bdi.min(),
			end = bdi.max())
		s = s.reindex(index=cdi).fillna(0).astype(int).cumsum()
		dte = (datetime.datetime.strptime(option, "%Y-%m-%d") - today).days
		tdte = s[option] - s[today_str]
		years_to_expiry = dte/365.25
		print("{} {} {:>3}DTE ({} calendar days)".format(ticker, option, tdte, dte))

		chain = quote.option_chain(option)
		calls = chain.calls
		calls['mark'] = (calls['bid'] + calls['ask'])/2
		calls['ivol'] = calls.apply(lambda x: ImpliedVolatilityCall(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['mark']), axis = 1)
		calls['d1'] = calls.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivol']), axis=1)
		calls['d2'] = calls.apply(lambda x: d2(x['d1'], years_to_expiry, x['ivol']), axis=1)
		calls['delta'] = calls.apply(lambda x: delta_call(x['d1'], years_to_expiry, dividend_yield), axis=1)
		calls['leverage'] = calls['delta'] * stock_price / calls['mark']
		print(calls[['contractSymbol', 'delta']])

		puts = chain.puts
		puts['mark'] = (puts['bid'] + puts['ask'])/2
		puts['ivol'] = puts.apply(lambda x: ImpliedVolatilityPut(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['mark']), axis = 1)
		puts['d1'] = puts.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivol']), axis=1)
		puts['d2'] = puts.apply(lambda x: d2(x['d1'], years_to_expiry, x['ivol']), axis=1)
		puts['delta'] = puts.apply(lambda x: delta_put(x['d1'], years_to_expiry, dividend_yield), axis=1)
		puts['leverage'] = puts['delta'] * stock_price / puts['mark']
		print(puts[['contractSymbol', 'delta']])
