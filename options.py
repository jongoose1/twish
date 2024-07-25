import yfinance as yf
import datetime
from zoneinfo import ZoneInfo
import time
import numpy as np
from scipy import optimize
from scipy.stats import norm

def ZCB(risk_free_rate, years_to_expiry):
	return (1+risk_free_rate)**(-years_to_expiry)

def d1(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility):
	return (np.log(stock_price/strike_price) + (risk_free_rate-dividend_yield+volatility**2/2)*years_to_expiry) / (volatility*np.sqrt(years_to_expiry))

def d2(d_1, years_to_expiry, volatility):
	return d_1 - volatility * np.sqrt(years_to_expiry)

def BSM_CALL(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility):
	if volatility == 0:
		return max(stock_price*(1+risk_free_rate)**years_to_expiry - strike_price, 0)
	d_1 = d1(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility)
	return stock_price * np.exp(-dividend_yield*years_to_expiry) * norm.cdf(d_1) - strike_price * np.exp(-risk_free_rate*years_to_expiry) * norm.cdf(d2(d_1, years_to_expiry, volatility))

def BSM_PUT(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility):
	if volatility == 0:
		return max(strike_price - stock_price*(1+risk_free_rate)**years_to_expiry, 0)
	d_1 = d1(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility)
	return strike_price * np.exp(-risk_free_rate*years_to_expiry) * norm.cdf(-d2(d_1, years_to_expiry, volatility)) - stock_price * np.exp(-dividend_yield*years_to_expiry) * norm.cdf(-d_1)

def call_iv_error(iv_guess, *args):
	stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, call_price = args
	return (BSM_CALL(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, iv_guess)-call_price)**2

def put_iv_error(iv_guess, *args):
	stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, put_price = args
	return (BSM_PUT(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, iv_guess)-put_price)**2

def ImpliedVolatilityCall(guess, *args):
	if guess == 0:
		guess = 0.3
	return optimize.fmin(func=call_iv_error, x0=guess, args=args, disp=False)[0]

def ImpliedVolatilityPut(guess, *args):
	if guess == 0:
		guess = 0.3
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
	return (datetime.datetime.strptime(option, "%Y-%m-%d").replace(hour=16, tzinfo=ZoneInfo('America/New_York')).timestamp() - time.time()) / 86400

def get_rfr():
	try:
		risk_free_rate = yf.Ticker("^IRX").info['dayLow']/100
	except:
		print("Using fake risk free rate")
		risk_free_rate = 0.05
	return risk_free_rate

def add_custom_columns(chain, stock_price, years_to_expiry, risk_free_rate, dividend_yield):
	zcb = ZCB(risk_free_rate, years_to_expiry)
	calls = chain.calls
	calls['mark'] 	= (calls['bid'] + calls['ask'])/2
	calls['spread'] = (calls['ask'] - calls['bid'])/calls['mark']
	calls['ivol'] 	= calls.apply(lambda x: ImpliedVolatilityCall(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['mark']), axis = 1)
	calls['d1'] 	= calls.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivol']), axis=1)
	calls['d2'] 	= calls.apply(lambda x: d2(x['d1'], years_to_expiry, x['ivol']), axis=1)
	calls['delta'] 	= calls.apply(lambda x: delta_call(x['d1'], years_to_expiry, dividend_yield), axis=1)
	calls['omega'] 	= calls['delta'] * stock_price / calls['ask']
	calls['gamma'] 	= calls.apply(lambda x: gamma(x['d1'], stock_price, years_to_expiry, dividend_yield, x['ivol']), axis=1)
	calls['theta'] 	= calls.apply(lambda x: theta_call(x['d1'], x['d2'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivol']), axis=1)
	calls['vega'] 	= calls.apply(lambda x: vega(x['d1'], stock_price, years_to_expiry, dividend_yield, x['ivol']), axis=1)
	calls['rho'] 	= calls.apply(lambda x: rho_call(x['d2'], x['strike'], years_to_expiry, risk_free_rate), axis=1)
	calls['iprob'] 	= calls.apply(lambda x: ImpliedProbabilityCall(x['d2']), axis=1)
	calls['ivolBid'] 	= calls.apply(lambda x: ImpliedVolatilityCall(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['bid']), axis = 1)
	calls['d1Bid'] 		= calls.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivolBid']), axis=1)
	calls['d2Bid'] 		= calls.apply(lambda x: d2(x['d1Bid'], years_to_expiry, x['ivolBid']), axis=1)
	calls['iprobBid'] 	= calls.apply(lambda x: ImpliedProbabilityCall(x['d2Bid']), axis=1)
	calls['ivolAsk'] 	= calls.apply(lambda x: ImpliedVolatilityCall(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ask']), axis = 1)
	calls['d1Ask'] 		= calls.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivolAsk']), axis=1)
	calls['d2Ask'] 		= calls.apply(lambda x: d2(x['d1Ask'], years_to_expiry, x['ivolAsk']), axis=1)
	calls['iprobAsk'] 	= calls.apply(lambda x: ImpliedProbabilityCall(x['d2Ask']), axis=1)
	calls['%TM'] 		= (calls['strike'] - stock_price) / stock_price
	calls['intrinsic'] 	= calls.apply(lambda x: max(stock_price - x['strike'], 0), axis=1)
	calls['extrinsic'] 	= calls['mark'] - calls['intrinsic']
	for i in range(0, len(calls)-1):
		#increasing strike
		calls.loc[i, 'bid_spread'] = calls.loc[i, 'bid'] - calls.loc[i+1, 'ask']
		calls.loc[i, 'ask_spread'] = calls.loc[i, 'ask'] - calls.loc[i+1, 'bid']
		calls.loc[i, 'delta_spread'] = calls.loc[i, 'delta'] - calls.loc[i+1, 'delta']
		calls.loc[i, 'theta_spread'] = calls.loc[i, 'theta'] - calls.loc[i+1, 'theta']
		calls.loc[i, 'iprobBid_spread'] = calls.loc[i, 'bid_spread'] / ((calls.loc[i+1, 'strike'] - calls.loc[i, 'strike'])*zcb)
		calls.loc[i, 'iprobAsk_spread'] = calls.loc[i, 'ask_spread'] / ((calls.loc[i+1, 'strike'] - calls.loc[i, 'strike'])*zcb)
	calls.loc[len(calls), 'bid_spread'] = float('nan')
	calls.loc[len(calls), 'ask_spread'] = float('nan')
	calls['omega_spread'] = calls['delta_spread'] * stock_price / calls['ask_spread']

	puts = chain.puts
	puts['mark'] 	= (puts['bid'] + puts['ask'])/2
	puts['spread'] 	= (puts['ask'] - puts['bid'])/puts['mark']
	puts['ivol'] 	= puts.apply(lambda x: ImpliedVolatilityPut(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['mark']), axis = 1)
	puts['d1'] 		= puts.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivol']), axis=1)
	puts['d2'] 		= puts.apply(lambda x: d2(x['d1'], years_to_expiry, x['ivol']), axis=1)
	puts['delta'] 	= puts.apply(lambda x: delta_put(x['d1'], years_to_expiry, dividend_yield), axis=1)
	puts['omega'] 	= puts['delta'] * stock_price / puts['ask']
	puts['gamma'] 	= puts.apply(lambda x: gamma(x['d1'], stock_price, years_to_expiry, dividend_yield, x['ivol']), axis=1)
	puts['theta'] 	= puts.apply(lambda x: theta_put(x['d1'], x['d2'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivol']), axis=1)
	puts['vega'] 	= puts.apply(lambda x: vega(x['d1'], stock_price, years_to_expiry, dividend_yield, x['ivol']), axis=1)
	puts['rho'] 	= puts.apply(lambda x: rho_put(x['d2'], x['strike'], years_to_expiry, risk_free_rate), axis=1)
	puts['iprob'] 	= puts.apply(lambda x: ImpliedProbabilityPut(x['d2']), axis=1)
	puts['ivolBid'] 	= puts.apply(lambda x: ImpliedVolatilityPut(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['bid']), axis = 1)
	puts['d1Bid'] 		= puts.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivolBid']), axis=1)
	puts['d2Bid'] 		= puts.apply(lambda x: d2(x['d1Bid'], years_to_expiry, x['ivolBid']), axis=1)
	puts['iprobBid'] 	= puts.apply(lambda x: ImpliedProbabilityPut(x['d2Bid']), axis=1)
	puts['ivolAsk'] 	= puts.apply(lambda x: ImpliedVolatilityPut(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ask']), axis = 1)
	puts['d1Ask'] 		= puts.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivolAsk']), axis=1)
	puts['d2Ask'] 		= puts.apply(lambda x: d2(x['d1Ask'], years_to_expiry, x['ivolAsk']), axis=1)
	puts['iprobAsk'] 	= puts.apply(lambda x: ImpliedProbabilityPut(x['d2Ask']), axis=1)
	puts['intrinsic'] 	= puts.apply(lambda x:  max(x['strike'] - stock_price, 0), axis=1)
	puts['extrinsic'] 	= puts['mark'] - puts['intrinsic']
	puts['%TM'] 		= (puts['strike'] - stock_price) / stock_price
	puts.loc[0, 'spreadBid'] = float('nan')
	puts.loc[0, 'spreadAsk'] = float('nan')
	for i in range(1, len(puts)):
		puts.loc[i, 'bid_spread'] = puts.loc[i, 'bid'] - puts.loc[i-1, 'ask']
		puts.loc[i, 'ask_spread'] = puts.loc[i, 'ask'] - puts.loc[i-1, 'bid']
		puts.loc[i, 'delta_spread'] = puts.loc[i, 'delta'] - puts.loc[i-1, 'delta']
		puts.loc[i, 'theta_spread'] = puts.loc[i, 'theta'] - puts.loc[i-1, 'theta']
		puts.loc[i, 'iprobBid_spread'] = puts.loc[i, 'bid_spread'] / ((puts.loc[i, 'strike'] - puts.loc[i-1, 'strike'])*zcb)
		puts.loc[i, 'iprobAsk_spread'] = puts.loc[i, 'ask_spread'] / ((puts.loc[i, 'strike'] - puts.loc[i-1, 'strike'])*zcb)
	puts['omega_spread'] = puts['delta_spread'] * stock_price / puts['ask_spread']

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
