import yfinance as yf
import datetime
from zoneinfo import ZoneInfo
import time
import numpy as np
from scipy import optimize
from scipy.stats import norm, t
import statistics
import pandas_market_calendars as mcal
import pandas as pd

def ZCB(risk_free_rate, years_to_expiry):
	return (1+risk_free_rate)**(-years_to_expiry)

def annualize(rate, days):
	return (1 + rate)**(252/days) - 1

def diemize(rate, days):
	return (1 + rate)**(1/days) - 1

def LTM_signal(bid, ask, ltm):
	if ask == 0:
		return 'N/A'
	if ltm > ask:
		return 'BUY'
	if ltm < bid:
		return 'SELL'
	return 'HOLD'

def p_ruin(nu,mu,tau, days=1):
	# probability of ruin after days days
	# = 1 - probability(survival)
	p = t.cdf(-100, nu, mu, tau)
	return 1 - (1 - p)**days

def kelly_stock(nu, mu, tau):
	p = 1 - t.cdf(0,nu,mu,tau)
	q = 1 - p
	g = t.expect(lambda x: x/100, (nu,), mu, tau, lb=0, conditional=True)
	l = t.expect(lambda x: -max(x,-100)/100, (nu,), mu, tau, ub=0, conditional=True)
	return p/l - q/g

def d1(stock_price, strike_price, years_to_expiry, risk_free_rate, dividend_yield, volatility):
	return (np.log(stock_price/strike_price) + (risk_free_rate-dividend_yield+volatility**2/2)*years_to_expiry) / (volatility*np.sqrt(years_to_expiry))

def d2(d_1, years_to_expiry, volatility):
	return d_1 - volatility * np.sqrt(years_to_expiry)

def ltm_next_prices(initial_prices, nu, mu, tau, additional_days):
	for x in range(additional_days):
		initial_prices = [price*(1+(mu+tau*np.random.standard_t(nu))/100) for price in initial_prices]
	return initial_prices

def ltm_final_prices(initial_price, nu,mu,tau,tdte_, trials):
	final_prices = [initial_price] * trials
	return ltm_next_prices(final_prices, nu, mu, tau, tdte_)

def LTM_CALL(final_prices, strike_price, zcb):
	payouts = [max(final_price-strike_price, 0) for final_price in final_prices]
	return statistics.fmean(payouts) * zcb

def LTM_PUT(final_prices, strike_price, zcb):
	payouts = [min(max(strike_price-final_price,0), strike_price) for final_price in final_prices]
	return statistics.fmean(payouts) * zcb

def ltm_cdf(final_prices, price):
	#p (price < final_price)
	return sum(1 for x in final_prices if x < price) / len(final_prices)

def ltm_p_gain_call(final_prices, strike, cost):
	return sum(1 for x in final_prices if x > strike+cost) / len(final_prices)

def ltm_kelly_call(final_prices, strike, cost):
	if cost == 0:
		return float('nan')
	breakeven = strike + cost
	# p = p(gain)
	p = sum(1 for x in final_prices if x > breakeven) / len(final_prices)
	#q = 1 - p
	q = 1 - p
	if (p == 0):
		return 0
	if (q == 0):
		return 1
	#g = fraction gained on positive outcome
	g = statistics.fmean([(x-breakeven)/cost for x in final_prices if x > breakeven])
	#l = fraction that is lost on negative outcome
	l = statistics.fmean([(cost - max(x-strike, 0))/cost for x in final_prices if x < breakeven])
	return min(p/l - q/g, 1)

def ltm_p_gain_put(final_prices, strike, cost):
	return sum(1 for x in final_prices if x < strike-cost) / len(final_prices)

def ltm_kelly_put(final_prices, strike, cost):
	if cost == 0:
		return float('nan')
	breakeven = strike - cost
	# p = p(gain)
	p = sum(1 for x in final_prices if x < breakeven) / len(final_prices)
	#q = 1 - p
	q = 1 - p
	if (p == 0):
		return 0
	if (q == 0):
		return 1
	#g = fraction gained on positive outcome
	g = statistics.fmean([(min(max(strike-x,0),strike)-cost)/cost for x in final_prices if x < breakeven])
	#l = fraction that is lost on negative outcome
	l = statistics.fmean([(cost - max(strike-x, 0))/cost for x in final_prices if x > breakeven])
	return min(p/l - q/g, 1)


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

today_str = datetime.date.today().strftime("%Y-%m-%d")
far_future_str = (datetime.date.today() + datetime.timedelta(days=1825)).strftime("%Y-%m-%d")
cal = mcal.get_calendar("NYSE")
bdi = mcal.date_range(cal.schedule(start_date = today_str, end_date = far_future_str), frequency = '1D')
bdi = bdi.normalize()
s = pd.Series(data = 1, index = bdi)
cdi = pd.date_range(start = bdi.min(),end = bdi.max())
s = s.reindex(index = cdi).fillna(0).astype(int).cumsum()
def tdte(option):
	# option str %Y-%m-%d
	return s[option]

def get_rfr():
	try:
		risk_free_rate = yf.Ticker("^IRX").info['dayLow']/100
	except:
		print("Using fake risk free rate")
		risk_free_rate = 0.05
	return risk_free_rate

def add_custom_columns(chain, stock_price, years_to_expiry, risk_free_rate, dividend_yield, nu, mu, tau, trials, tdte_, final_prices = None, bsm=True, ltm=True, spread=True, fly=True):
	zcb = ZCB(risk_free_rate, years_to_expiry)
	rfg = 1 / zcb - 1
	if (final_prices==None):
		final_prices = ltm_final_prices(stock_price, nu, mu, tau, tdte_, trials)
	calls = chain.calls
	calls['mark'] 	= (calls['bid'] + calls['ask'])/2
	calls['spread'] = (calls['ask'] - calls['bid'])/calls['mark']
	if(bsm):
		calls['ivol'] 	= calls.apply(lambda x: ImpliedVolatilityCall(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['mark']), axis = 1)
		calls['d1'] 	= calls.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivol']), axis=1)
		calls['d2'] 	= calls.apply(lambda x: d2(x['d1'], years_to_expiry, x['ivol']), axis=1)
		calls['delta'] 	= calls.apply(lambda x: delta_call(x['d1'], years_to_expiry, dividend_yield), axis=1)
		calls['omega'] 	= calls['delta'] * stock_price / calls['ask']
		calls['gamma'] 	= calls.apply(lambda x: gamma(x['d1'], stock_price, years_to_expiry, dividend_yield, x['ivol']), axis=1)
		calls['theta'] 	= calls.apply(lambda x: theta_call(x['d1'], x['d2'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivol']), axis=1)
		calls['vega'] 	= calls.apply(lambda x: vega(x['d1'], stock_price, years_to_expiry, dividend_yield, x['ivol']), axis=1)
		calls['rho'] 	= calls.apply(lambda x: rho_call(x['d2'], x['strike'], years_to_expiry, risk_free_rate), axis=1)
		calls['BSM_CDF%'] 	= 100 - 100 * calls.apply(lambda x: ImpliedProbabilityCall(x['d2']), axis=1)
		calls['ivolBid'] 	= calls.apply(lambda x: ImpliedVolatilityCall(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['bid']), axis = 1)
		calls['d1Bid'] 		= calls.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivolBid']), axis=1)
		calls['d2Bid'] 		= calls.apply(lambda x: d2(x['d1Bid'], years_to_expiry, x['ivolBid']), axis=1)
		calls['iprobBid'] 	= calls.apply(lambda x: ImpliedProbabilityCall(x['d2Bid']), axis=1)
		calls['ivolAsk'] 	= calls.apply(lambda x: ImpliedVolatilityCall(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ask']), axis = 1)
		calls['d1Ask'] 		= calls.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivolAsk']), axis=1)
		calls['d2Ask'] 		= calls.apply(lambda x: d2(x['d1Ask'], years_to_expiry, x['ivolAsk']), axis=1)
		calls['iprobAsk'] 	= calls.apply(lambda x: ImpliedProbabilityCall(x['d2Ask']), axis=1)
	calls['%TM'] 		= 100 * (calls['strike'] - stock_price) / stock_price
	calls['breakeven'] 	= calls['strike'] + calls['ask']
	calls['%BE'] 		= 100 * (calls['breakeven'] - stock_price) / stock_price
	calls['intrinsic'] 	= calls.apply(lambda x: max(stock_price - x['strike'], 0), axis=1)
	calls['extrinsic'] 	= calls['mark'] - calls['intrinsic']

	#LTM
	if(ltm):
		calls['LTM']		= calls.apply(lambda x: LTM_CALL(final_prices, x['strike'], zcb), axis=1)
		calls['LTM_payout'] = calls['LTM'] / zcb
		calls['LTM_E']		= calls['LTM_payout'] - calls['ask']
		calls['LTM_E%']		= 100 * calls['LTM_E'] / calls['ask']
		calls['LTM_signal'] = calls.apply(lambda x: LTM_signal(x['bid'], x['ask'], x['LTM']), axis=1)
		calls['Markup%']	= 100 * (calls['ask'] - calls['LTM']) /  calls['LTM']
		calls['Kelly'] 	= calls.apply(lambda x: ltm_kelly_call(final_prices, x['strike'], x['ask']), axis=1)
		calls['Kelly%'] 	= 100 * calls['Kelly']
		calls['P(gain)%'] 	= calls.apply(lambda x: 100*ltm_p_gain_call(final_prices, x['strike'], x['ask']), axis=1)
		calls['LTM_CDF%']	= calls.apply(lambda x: 100*ltm_cdf(final_prices, x['strike']), axis=1)
		calls['LTM_KE%']	= calls.apply(lambda x: max(x['LTM_E%'], 0)*max(x['Kelly'], 0), axis=1) #ignore negative*negative values for now
		calls['LTM_KG%']	= calls['LTM_KE%'] + 100 * (1 - calls['Kelly']) * rfg
		calls['LTM_KGD%']	= calls.apply(lambda x: 100 * diemize(x['LTM_KG%']/100, tdte_), axis=1)

	#spread
	if(spread):
		for i in range(0, len(calls)-1):
			#increasing strike
			calls.loc[i, 'bid_spread'] = calls.loc[i, 'bid'] - calls.loc[i+1, 'ask']
			calls.loc[i, 'ask_spread'] = calls.loc[i, 'ask'] - calls.loc[i+1, 'bid']
			if(bsm):
				calls.loc[i, 'delta_spread'] = calls.loc[i, 'delta'] - calls.loc[i+1, 'delta']
				calls.loc[i, 'theta_spread'] = calls.loc[i, 'theta'] - calls.loc[i+1, 'theta']
			calls.loc[i, 'width_spread'] = calls.loc[i+1, 'strike'] - calls.loc[i, 'strike']
			calls.loc[i, 'strike_spread'] = (calls.loc[i, 'strike'] + calls.loc[i+1, 'strike']) / 2
			if(bsm):
				calls.loc[i, 'BSM_PDF'] = (calls.loc[i+1, 'BSM_CDF%'] - calls.loc[i, 'BSM_CDF%']) / (100 * calls.loc[i, 'width_spread'])
		if(bsm):
			calls['omega_spread'] = calls['delta_spread'] * stock_price / calls['ask_spread']
		calls['mark_spread'] = (calls['bid_spread'] + calls['ask_spread']) / 2
		#dC/dK = ck+h - ck / h = -spread/h
		#cdf = 1 + dC/dK * 1/z
		calls['DK'] = -calls['mark_spread'] / calls['width_spread'] #interpret at average of strikes. (strike_spread)
		calls['CDF%_spread'] = 100 * (1 + calls['DK'] / zcb)

		#fly
		if(fly):
			for i in range(0,len(calls)-2):
				calls.loc[i, 'bid_fly'] = calls.loc[i, 'bid_spread'] - calls.loc[i+1, 'ask_spread']
				calls.loc[i, 'ask_fly'] = calls.loc[i, 'ask_spread'] - calls.loc[i+1, 'bid_spread']
				calls.loc[i, 'width_fly'] = calls.loc[i+1, 'strike_spread'] - calls.loc[i, 'strike_spread']
				calls.loc[i, 'strike_fly'] = (calls.loc[i, 'strike_spread'] + calls.loc[i+1, 'strike_spread']) / 2
				calls.loc[i, 'DK2'] = (calls.loc[i+1, 'DK'] - calls.loc[i, 'DK']) / calls.loc[i, 'width_fly']
			calls['mark_fly'] = (calls['bid_fly'] + calls['ask_fly']) / 2
			calls['PDF_fly'] = 100 * calls['DK2'] / zcb
	
	puts = chain.puts
	puts['mark'] 	= (puts['bid'] + puts['ask'])/2
	puts['spread'] 	= (puts['ask'] - puts['bid'])/puts['mark']
	if(bsm):
		puts['ivol'] 	= puts.apply(lambda x: ImpliedVolatilityPut(x['impliedVolatility'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['mark']), axis = 1)
		puts['d1'] 		= puts.apply(lambda x: d1(stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivol']), axis=1)
		puts['d2'] 		= puts.apply(lambda x: d2(x['d1'], years_to_expiry, x['ivol']), axis=1)
		puts['delta'] 	= puts.apply(lambda x: delta_put(x['d1'], years_to_expiry, dividend_yield), axis=1)
		puts['omega'] 	= puts['delta'] * stock_price / puts['ask']
		puts['gamma'] 	= puts.apply(lambda x: gamma(x['d1'], stock_price, years_to_expiry, dividend_yield, x['ivol']), axis=1)
		puts['theta'] 	= puts.apply(lambda x: theta_put(x['d1'], x['d2'], stock_price, x['strike'], years_to_expiry, risk_free_rate, dividend_yield, x['ivol']), axis=1)
		puts['vega'] 	= puts.apply(lambda x: vega(x['d1'], stock_price, years_to_expiry, dividend_yield, x['ivol']), axis=1)
		puts['rho'] 	= puts.apply(lambda x: rho_put(x['d2'], x['strike'], years_to_expiry, risk_free_rate), axis=1)
		puts['BSM_CDF%'] 	= 100 * puts.apply(lambda x: ImpliedProbabilityPut(x['d2']), axis=1)
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
	puts['%TM'] 		= 100 * (puts['strike'] - stock_price) / stock_price
	puts['breakeven'] 	= puts['strike'] - puts['ask']
	puts['%BE'] 		= 100 * (puts['breakeven'] - stock_price) / stock_price
	
	#LTM
	if(ltm):
		puts['LTM']			= puts.apply(lambda x: LTM_PUT(final_prices, x['strike'], zcb), axis=1)
		puts['LTM_payout'] 	= puts['LTM'] / zcb
		puts['LTM_E']		= puts['LTM_payout'] - puts['ask']
		puts['LTM_E%']		= 100 * puts['LTM_E'] / puts['ask']
		puts['LTM_signal']  = puts.apply(lambda x: LTM_signal(x['bid'], x['ask'], x['LTM']), axis=1)
		puts['Markup%']		= 100 * (puts['ask'] - puts['LTM']) /  puts['LTM']
		puts['Kelly'] 		= puts.apply(lambda x: ltm_kelly_put(final_prices, x['strike'], x['ask']), axis=1)
		puts['Kelly%'] 		= 100 * puts['Kelly']
		puts['P(gain)%'] 	= puts.apply(lambda x: 100*ltm_p_gain_put(final_prices, x['strike'], x['ask']), axis=1)
		puts['LTM_CDF%']	= puts.apply(lambda x: 100*ltm_cdf(final_prices, x['strike']), axis=1)
		puts['LTM_KE%']		= puts.apply(lambda x: max(x['LTM_E%'], 0)*max(x['Kelly'], 0), axis=1) #ignore negative*negative values for now
		puts['LTM_KG%']		= puts['LTM_KE%'] + 100 * (1 - puts['Kelly']) * rfg
		puts['LTM_KGD%']	= puts.apply(lambda x: 100 * diemize(x['LTM_KG%']/100, tdte_), axis=1)
	
	#spread
	if(spread):
		for i in range(1, len(puts)):
			puts.loc[i, 'bid_spread'] = puts.loc[i, 'bid'] - puts.loc[i-1, 'ask']
			puts.loc[i, 'ask_spread'] = puts.loc[i, 'ask'] - puts.loc[i-1, 'bid']
			if(bsm):
				puts.loc[i, 'delta_spread'] = puts.loc[i, 'delta'] - puts.loc[i-1, 'delta']
				puts.loc[i, 'theta_spread'] = puts.loc[i, 'theta'] - puts.loc[i-1, 'theta']
			puts.loc[i, 'width_spread'] = puts.loc[i, 'strike'] - puts.loc[i-1, 'strike']
			puts.loc[i, 'strike_spread'] = (puts.loc[i, 'strike'] + puts.loc[i-1, 'strike']) / 2
			if(bsm):
				puts.loc[i, 'BSM_PDF'] = (puts.loc[i, 'BSM_CDF%'] - puts.loc[i-1, 'BSM_CDF%']) / (100 * puts.loc[i, 'width_spread'])
		if(bsm):
			puts['omega_spread'] = puts['delta_spread'] * stock_price / puts['ask_spread']
		puts['mark_spread'] = (puts['bid_spread'] + puts['ask_spread']) / 2
		#dP/dK = pk - pk-h / h = spread/h
		#cdf = dP/dK * 1/z
		puts['DK'] = puts['mark_spread'] / puts['width_spread'] #interpret at average of strikes. (strike_spread)
		puts['CDF%_spread'] = 100 * puts['DK'] / zcb

		#fly
		if(fly):
			for i in range(2,len(puts)):
				puts.loc[i, 'bid_fly'] = puts.loc[i, 'bid_spread'] - puts.loc[i-1, 'ask_spread']
				puts.loc[i, 'ask_fly'] = puts.loc[i, 'ask_spread'] - puts.loc[i-1, 'bid_spread']
				puts.loc[i, 'width_fly'] = puts.loc[i, 'strike_spread'] - puts.loc[i-1, 'strike_spread']
				puts.loc[i, 'strike_fly'] = (puts.loc[i, 'strike_spread'] + puts.loc[i-1, 'strike_spread']) / 2
				puts.loc[i, 'DK2'] = (puts.loc[i, 'DK'] - puts.loc[i-1, 'DK']) / puts.loc[i, 'width_fly']
			puts['mark_fly'] = (puts['bid_fly'] + puts['ask_fly']) / 2
			puts['PDF_fly'] = 100 * puts['DK2'] / zcb


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
