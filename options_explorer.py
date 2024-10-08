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
cols = ['bid', 'ask', 'omega', 'delta', 'theta', 'ivolBid', 'ivolAsk', 'BSM_CDF%', 'BSM_PDF', '%TM', 'breakeven', '%BE']
put_cols = ['strike'] + [col + '_put' for col in cols]
call_cols = ['strike'] + [col + '_call' for col in cols]
synthetic_cols = ['strike', 'IEST_LONG', 'C-P','IEST_SHORT', 'P-C']

spread_cols = ['strike', 'bid', 'mark','ask', 'omega', 'delta', 'theta', 'CDF%']
call_spread_cols = ['strike'] + [col+'_spread_call' for col in spread_cols]
put_spread_cols = ['strike'] + [col+'_spread_put' for col in spread_cols]

fly_cols = ['strike', 'bid', 'mark', 'ask', 'PDF']
call_fly_cols = ['strike'] + [col+'_fly_call' for col in fly_cols]
put_fly_cols = ['strike'] + [col+'_fly_put' for col in fly_cols]

ltm_cols = ['ask', 'LTM', 'LTM_signal', 'Markup%', 'Kelly%', 'P(gain)%', 'Avg(gain)%', 'Avg(loss)%', 'LTM_CDF%', 'LTM_E%', 'LTM_KGD%']
call_ltm_cols = ['strike'] + [col+'_call' for col in ltm_cols]
put_ltm_cols = ['strike'] + [col+'_put' for col in ltm_cols]
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
	print("T fit: DF={:.6f}, LOC={:.6f}, SCALE={:.6f}, MSE={:.6f}".format(df, loc, scale, mse))
	print("Kelly: {:.2f}%, P(ruin): {}".format(100*kelly_stock(df, loc, scale), p_ruin(df,loc,scale,1)))
	for option in options:
		print("{:<3} {} ({}DTE)	({} trading days) CALLS/PUTS".format(i, option, int(dte(option)), tdte(option)))
		i += 1

	input_ = input("Type the number of a chain and 'C' or 'P': ")
	index = int(regex.sub('', input_)) - 1
	option = options[index]
	if 'P' in input_.upper():
		print("PUT")
		columns_to_print = put_cols
	else:
		print("CALL")
		columns_to_print = call_cols
	
	years_to_expiry = dte(option)/365.25
	
	chain = quote.option_chain(option)
	
	add_custom_columns(chain, stock_price, years_to_expiry, risk_free_rate, dividend_yield, df, loc, scale, 300000,tdte(option))
	merged = chain.calls.merge(chain.puts, left_on='strike', right_on='strike', suffixes=('_call','_put'))
	merged['C-P'] = merged['ask_call']-merged['bid_put']
	merged['P-C'] = merged['bid_call']-merged['ask_put']
	merged['IEST_LONG'] = merged['strike'] + merged['C-P'] / ZCB(risk_free_rate, years_to_expiry)
	merged['IEST_SHORT'] = merged['strike'] + merged['P-C'] / ZCB(risk_free_rate, years_to_expiry)

	done = False
	while(not done):
		print("{}{:5} ${:<7.2f} {:<+3.2f}%{}".format(color, ticker, stock_price, percent_change, Fore.RESET, dividend_yield, risk_free_rate))
		print("Yield: {:.2f}%, Risk Free Rate {:.2f}%".format(100*dividend_yield, 100*risk_free_rate))
		print("T fit: DF={:.6f}, LOC={:.6f}, SCALE={:.6f}, MSE={:.6f}".format(df, loc, scale, mse))
		print("Kelly: {:.2f}%, P(ruin): {}".format(100*kelly_stock(df, loc, scale), p_ruin(df,loc,scale,1)))
		print("{} {} ({}DTE) ({} trading days)".format(ticker, option, int(dte(option)), tdte(option)))
		print(merged[columns_to_print].iloc[::-1].to_string(index=False))

		for col in merged.columns:
			if col not in columns_to_print:
				print(col, end=' ')
		print()
		columns_to_toggle = input("Toggle columns or put or call or synthetic or (c/p)(spread/fly/ltm) or done:")
		for col in columns_to_toggle.split():
			if col.upper() == 'DONE':
				done = True
				break
			elif col.upper() == 'PUT':
				print("PUT")
				columns_to_print = put_cols
				break
			elif col.upper() == 'CALL':
				print("CALL")
				columns_to_print = call_cols
				break
			elif col.upper() == 'SYNTHETIC':
				print("SYNTHETIC")
				columns_to_print = synthetic_cols
			elif col.upper() == 'CSPREAD':
				print("CALL SPREAD, 1 Strike Wide, 'strike' indicates long strike")
				columns_to_print = call_spread_cols
				break
			elif col.upper() == 'PSPREAD':
				print("PUT SPREAD, 1 Strike Wide, 'strike' indicates long strike")
				columns_to_print = put_spread_cols
				break
			elif col.upper() == 'PFLY':
				print("PUT FLY, 1 Strike Wide, 'strike' indicates most expensive long strike")
				columns_to_print = put_fly_cols
				break
			elif col.upper() == 'CFLY':
				print("CALL FLY, 1 Strike Wide, 'strike' indicates most expensive long strike")
				columns_to_print = call_fly_cols
				break
			elif col.upper() == 'CLTM':
				print('CLTM')
				columns_to_print = call_ltm_cols
				break
			elif col.upper() == 'PLTM':
				print('PLTM')
				columns_to_print = put_ltm_cols
				break
			elif col in merged.columns:
				if col in columns_to_print:
					columns_to_print.remove(col)
				else:
					columns_to_print.append(col)
