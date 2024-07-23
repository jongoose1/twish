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
		print("{:<3} {} ({}DTE)	CALLS/PUTS".format(i, option, int(dte_)))
		i += 1

	selection_id = input("Select: ")
	option = options[int(selection_id) - 1]
	
	years_to_expiry = dte(option)/365.25
	
	chain = quote.option_chain(option)
	add_custom_columns(chain, stock_price, years_to_expiry, risk_free_rate, dividend_yield)

	done = False
	while(not done):
		print("{}{:5} ${:<7.2f} {:<+3.2f}%{}".format(color, ticker, stock_price, percent_change, Fore.RESET, dividend_yield, risk_free_rate))
		print("Yield: {:.2f}%, Risk Free Rate {:.2f}%".format(100*dividend_yield, 100*risk_free_rate))
		print("{} {} CALLS {}DTE".format(ticker, option, int(dte(option))))
		print(chain.calls[columns_to_print].iloc[::-1].to_string(index=False))
	
		print("{}{:5} ${:<7.2f} {:<+3.2f}%{}".format(color, ticker, stock_price, percent_change, Fore.RESET, dividend_yield, risk_free_rate))
		print("Yield: {:.2f}%, Risk Free Rate {:.2f}%".format(100*dividend_yield, 100*risk_free_rate))
		print("{} {} PUTS {}DTE".format(ticker, option, int(dte(option))))
		print(chain.puts[columns_to_print].iloc[::-1].to_string(index=False))

		for col in chain.calls.columns:
			if col not in columns_to_print:
				print(col, end=' ')
		print()
		columns_to_toggle = input("Toggle columns or done:")
		for col in columns_to_toggle.split():
			if col.upper() == 'DONE':
				done = True
				break
			if col in chain.calls.columns:
				if col in columns_to_print:
					columns_to_print.remove(col)
				else:
					columns_to_print.append(col)
