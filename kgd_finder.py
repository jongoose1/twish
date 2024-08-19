import yfinance as yf
import logging
from colorama import Fore
import pandas as pd
from options import *
import re
from scipy import stats

logging.getLogger('yfinance').disabled = True
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f'%x)
regex = re.compile('[^0-9]')

risk_free_rate = get_rfr()
columns_to_print = ['exp', 'TDTE','strike', 'type', 'bid', 'ask', 'LTM', '%TM', '%BE','LTM_signal', 'Markup%', 'Kelly%', 'P(gain)%', 'LTM_CDF%', 'LTM_E%', 'LTM_KGD%']
while(True):
	ticker = input("Ticker or done: ").upper()
	if ticker == "DONE":
		exit()
	quote = yf.Ticker(ticker)
	info = quote.info
	if 'symbol' not in info:
		print(info)
		print("invalid ticker")
		continue
	
	fit = False
	while (not fit):
		sub = input("substitue t fit? (y/n/manual): ").upper()
		if sub == 'Y' or sub=='YES':
			new_ticker = input("substitue ticker: ")
			fit, df, loc, scale, mse = get_t_fit(new_ticker)
		elif sub == 'M' or sub == 'MANUAL':
			fit = True
			df = float(input("NU/DF: "))
			loc = float(input("MU/LOC: "))
			scale = float(input("TAU/SCALE: "))
		else:
			fit, df, loc, scale, mse = get_t_fit(ticker)
	
	max_tdte = int(input("max tdte:"))
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
	print("{}{:5} ${:<7.2f} {:<+3.2f}%{}".format(color, ticker, stock_price, percent_change, Fore.RESET, dividend_yield, risk_free_rate))
	print("Yield: {:.2f}%, Risk Free Rate {:.2f}%".format(100*dividend_yield, 100*risk_free_rate))
	print("T fit: DF={:.6f}, LOC={:.6f}, SCALE={:.6f}".format(df, loc, scale))
	print("Kelly: {:.2f}%, P(ruin): {}".format(100*kelly_stock(df, loc, scale), p_ruin(df,loc,scale,1)))

	
	all_list = []
	trials = 100000
	final_prices = [stock_price] * trials
	prev_tdte = 0
	for option in options:
		tdte_ = tdte(option)
		if (tdte_ > max_tdte):
			continue
		final_prices = ltm_next_prices(final_prices, df, loc, scale, tdte_-prev_tdte)
		print("Calculating metrics for options expiring {}".format(option))
		years_to_expiry = dte(option)/365.25
		chain = quote.option_chain(option)
		add_custom_columns(chain, stock_price, years_to_expiry, risk_free_rate, dividend_yield, df, loc, scale, trials,tdte(option), final_prices, bsm=False, spread=False)
		chain.calls['TDTE'] = tdte(option)
		chain.calls['type'] = 'CALL'
		chain.calls['exp'] = option
		chain.puts['TDTE'] = tdte(option)
		chain.puts['type'] = 'PUT'
		chain.puts['exp'] = option
		all_list.append(chain.calls)
		all_list.append(chain.puts)
		prev_tdte = tdte_

	all_options = pd.concat(all_list, axis=0).sort_values(by=['LTM_KGD%'], ascending=False)

	done = False
	while(not done):
		print("{}{:5} ${:<7.2f} {:<+3.2f}%{}".format(color, ticker, stock_price, percent_change, Fore.RESET, dividend_yield, risk_free_rate))
		print("Yield: {:.2f}%, Risk Free Rate {:.2f}%".format(100*dividend_yield, 100*risk_free_rate))
		print("T fit: DF={:.6f}, LOC={:.6f}, SCALE={:.6f}".format(df, loc, scale))
		print("Kelly%: {}, P(ruin): {}".format(100*kelly_stock(df, loc, scale), p_ruin(df,loc,scale,1)))
		print(all_options[columns_to_print].head(50).to_string(index=False))

		for col in all_options.columns:
			if col not in columns_to_print:
				print(col, end=' ')
		print()
		columns_to_toggle = input("Toggle columns or done:")
		for col in columns_to_toggle.split():
			if col.upper() == 'DONE':
				done = True
				break
			if col in all_options.columns:
				if col in columns_to_print:
					columns_to_print.remove(col)
				else:
					columns_to_print.append(col)
