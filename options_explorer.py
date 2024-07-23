import yfinance as yf
import logging
from colorama import Fore
import pandas as pd
from options import *

logging.getLogger('yfinance').disabled = True
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f'%x)

risk_free_rate = get_rfr()
columns_to_print = ['strike', 'bid', 'ask', 'omega', 'delta', 'theta', 'ivolBid', 'ivolAsk', 'iprob', '%TM', 'intrinsic', 'extrinsic']

while(True):
	ticker = input("Ticker or done: ").upper()
	if ticker == "DONE":
		exit()
	quote = yf.Ticker(ticker)
	info = quote.info
	if 'symbol' not in info:
		print("invalid ticker")
		continue
	options = quote.options
	stock_price = (info['bid'] + info['ask']) / 2 
	percent_change =100 * (stock_price - info['previousClose'])/ info['previousClose']
	if percent_change > 0:
		color = Fore.GREEN
	else:
		color = Fore.RED
	if 'yield' in info:
		dividend_yield = info['yield']
	else:
		dividend_yield = 0
	i = 1
	print("{}{:5} ${:<7.2f} {:<+3.2f}%{}".format(color, ticker, stock_price, percent_change, Fore.RESET, dividend_yield, risk_free_rate))
	print("Yield: {:.2f}%, Risk Free Rate {:.2f}%".format(100*dividend_yield, 100*risk_free_rate))
	for option in options:
		dte_ = dte(option)
		print("{:<3} {} ({}DTE)	CALLS/PUTS".format(i, option, dte_))
		i += 1

	selection_id = input("Select: ")
	option = options[int(selection_id) - 1]

	years_to_expiry = dte(option)/365.25
	chain = quote.option_chain(option)
	
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
	
	done = False
	while(not done):
		print("{}{:5} ${:<7.2f} {:<+3.2f}%{}".format(color, ticker, stock_price, percent_change, Fore.RESET, dividend_yield, risk_free_rate))
		print("Yield: {:.2f}%, Risk Free Rate {:.2f}%".format(100*dividend_yield, 100*risk_free_rate))
		print("{} {} CALLS {}DTE".format(ticker, option, dte(option)))
		print(calls[columns_to_print].iloc[::-1])
	
		print("{}{:5} ${:<7.2f} {:<+3.2f}%{}".format(color, ticker, stock_price, percent_change, Fore.RESET, dividend_yield, risk_free_rate))
		print("Yield: {:.2f}%, Risk Free Rate {:.2f}%".format(100*dividend_yield, 100*risk_free_rate))
		print("{} {} PUTS {}DTE".format(ticker, option, dte(option)))
		print(puts[columns_to_print].iloc[::-1])

		for col in calls.columns:
			if col not in columns_to_print:
				print(col, end=' ')
		print()
		columns_to_toggle = input("Toggle columns or done:")
		for col in columns_to_toggle.split():
			if col.upper() == 'DONE':
				done = True
				break
			if col in calls.columns:
				if col in columns_to_print:
					columns_to_print.remove(col)
				else:
					columns_to_print.append(col)
