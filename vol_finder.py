import yfinance as yf
import logging
from colorama import Fore
import pandas as pd
from options import *

logging.getLogger('yfinance').disabled = True
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f'%x)

risk_free_rate = get_rfr()
columns_to_print = ['dte', 'strike', 'bid', 'ask', 'omega', 'delta', 'theta', 'ivolBid', 'ivolAsk', 'iprob', '%TM']

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
	calls_list = []
	puts_list = []
	for option in options:
		years_to_expiry = dte(option)/365.25
		chain = quote.option_chain(option)
		add_custom_columns(chain, stock_price, years_to_expiry, risk_free_rate, dividend_yield)
		chain.calls['dte'] = int(dte(option))
		chain.puts['dte'] = int(dte(option))
		calls_list.append(chain.calls)
		puts_list.append(chain.puts)

	
	calls = pd.concat(calls_list, axis=0).sort_values(by=['ivolBid'], ascending=False)
	puts = pd.concat(puts_list, axis=0).sort_values(by=['ivolBid'], ascending=False)
	#calls = calls[(calls[['bid', 'ask', 'ivolBid', 'ivolAsk']] != 0).any(axis=1)]
	#puts = puts[(puts[['bid', 'ask', 'ivolBid', 'ivolAsk']] != 0).any(axis=1)]

	done = False
	while(not done):
		print("{}{:5} ${:<7.2f} {:<+3.2f}%{}".format(color, ticker, stock_price, percent_change, Fore.RESET, dividend_yield, risk_free_rate))
		print("Yield: {:.2f}%, Risk Free Rate {:.2f}%".format(100*dividend_yield, 100*risk_free_rate))
		print("CALLS")
		print(calls[columns_to_print].head(50).to_string(index=False))
		print("{}{:5} ${:<7.2f} {:<+3.2f}%{}".format(color, ticker, stock_price, percent_change, Fore.RESET, dividend_yield, risk_free_rate))
		print("Yield: {:.2f}%, Risk Free Rate {:.2f}%".format(100*dividend_yield, 100*risk_free_rate))
		print("PUTS")
		print(puts[columns_to_print].head(50).to_string(index=False))

		for col in calls.columns:
			if col not in columns_to_print:
				print(col, end=' ')
		print()
		columns_to_toggle = input("Toggle columns or or buy or sell or done:")
		for col in columns_to_toggle.split():
			if col.upper() == 'DONE':
				done = True
				break
			if col.upper() == 'SELL':
				calls = calls.sort_values(by=['ivolBid'], ascending=False)
				puts = puts.sort_values(by=['ivolBid'], ascending=False)
				#calls = calls[(calls[['bid', 'ask', 'ivolBid', 'ivolAsk']] != 0).any(axis=1)]
				#puts = puts[(puts[['bid', 'ask', 'ivolBid', 'ivolAsk']] != 0).any(axis=1)]
			if col.upper() == 'BUY':
				calls = calls.sort_values(by=['ivolAsk'], ascending=True)
				puts = puts.sort_values(by=['ivolAsk'], ascending=True)
				#calls = calls[(calls[['bid', 'ask', 'ivolBid', 'ivolAsk']] != 0).any(axis=1)]
				#puts = puts[(puts[['bid', 'ask', 'ivolBid', 'ivolAsk']] != 0).any(axis=1)]
			if col in calls.columns:
				if col in columns_to_print:
					columns_to_print.remove(col)
				else:
					columns_to_print.append(col)
