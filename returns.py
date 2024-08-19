import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import stats, fft, signal
import numpy as np
import sounddevice as sd
import time
import threading
from statsmodels.tsa.ar_model import AutoReg
#import statsmodels as sm
import statsmodels.api as sm
from arch.unitroot import ADF
from time import sleep

#ticker = input("ticker: ").upper()
ticker = "^SPX"
try:
	data = pd.read_pickle(ticker+'.pkl')
	print("Previous data found")
except:
	data = yf.download(ticker, period='max')
	if data.empty:
		exit()
	data.to_pickle(ticker+'.pkl')

data = data.reset_index()
alpha = 0.001
data.loc[0, 'abs_pc'] = 0 
ggain = 0
gloss = 0
ngains = 0
nnno = 0

data['prevClose'] = data['Close'].shift(1)
data['percent_change'] = 100*(data['Close']-data['prevClose'])/data['prevClose']
data.loc[0, 'percent_change'] = 1
data['abs_pc'] = data.apply(lambda x: abs(x['percent_change']), axis=1)

'''
data.loc[0, 'a'] = 1
for i in range(1, len(data)):
	data.loc[i, 'a'] = alpha*data.loc[i, 'abs_pc'] + (1-alpha) * data.loc[i-1, 'a']
data['n'] = (data['abs_pc'] - data['a'])/data['a']
data.loc[0, 'n'] = -1
nnno = sum(1 for n in data[1:]['n'] if n==-1)
'''

data.plot(y = 'percent_change')
data.plot(y = 'abs_pc')
#plt.plot(data['a'])
'''
plt.figure(5)
plt.plot(data['n'])
'''
plt.figure(3)
num_bins = 150
actuals, bins_lefts, patches = plt.hist(data['percent_change'], bins=num_bins)
xmin, xmax = plt.xlim()
#x = np.linspace(xmin, xmax, num_bins)
bin_width = (xmax-xmin)/num_bins
bin_centers = [bin_left+bin_width/2 for bin_left in bins_lefts]

#normal fit
mu, std = stats.norm.fit(data['percent_change'])
print("Normal fit: MU={:.6f}, STD={:.6f}".format(mu, std))
norm_pdf = stats.norm.pdf(bin_centers, mu, std)
normals = norm_pdf*bin_width*len(data)
plt.plot(bin_centers, normals, color='red', linewidth=1)

#cauchy fit
loc, scale = stats.cauchy.fit(data['percent_change'])
print("Cauchy fit: LOC={:.6f}, SCALE={:.6f}".format(loc, scale))
cauchy_pdf = stats.cauchy.pdf(bin_centers, loc, scale)
cauchies = cauchy_pdf*bin_width*len(data)
plt.plot(bin_centers, cauchies, color='yellow', linewidth=1)

#t fit
df, loc, scale = stats.t.fit(data['percent_change'])
print("T fit: DF={:.6f}, LOC={:.6f}, SCALE={:.6f}".format(df, loc, scale))
t_pdf = stats.t.pdf(bin_centers, df, loc, scale)
ts = t_pdf*bin_width*len(data)
plt.plot(bin_centers, ts, color='green', linewidth=1)

#LAPLACE FIT
loc, scale = stats.laplace.fit(data['percent_change'])
print("Laplace Fit: loc={:.6f}, scale={:.6f}".format(loc, scale))
laplace_pdf = stats.laplace.pdf(bin_centers, loc, scale)
laplaces = laplace_pdf*bin_width*len(data)
plt.plot(bin_centers, laplaces, color="orange", linewidth=1)

'''
plt.figure(4)
actuals, bins_lefts, patches = plt.hist(data['n'], bins=num_bins)
xmin, xmax = plt.xlim()
bin_width = (xmax-xmin)/num_bins
bin_centers = [bin_left+bin_width/2 for bin_left in bins_lefts]

loc, scale = stats.expon.fit(data['n'])
print("e fit: loc={:.6f}, scale={:.6f}".format(loc, scale))
e_pdf = stats.expon.pdf(bin_centers, loc, scale)
es = e_pdf*bin_width*len(data)
plt.plot(bin_centers, es, color = 'red', linewidth=1)

b, loc, scale = stats.pareto.fit(data['n'])
print("pareto fit: b={:.6f}, loc={:.6f}, scale={:.6f}".format(b, loc, scale))
pareto_pdf = stats.pareto.pdf(bin_centers,b, loc, scale)
paretos = pareto_pdf*bin_width*len(data)
plt.plot(bin_centers, paretos, color = 'green', linewidth=1)

df, loc, scale = stats.chi.fit(data['n'])
print("chi fit: b={:.6f}, loc={:.6f}, scale={:.6f}".format(df, loc, scale))
chi_pdf = stats.chi.pdf(bin_centers,df, loc, scale)
chis = chi_pdf*bin_width*len(data)
plt.plot(bin_centers, chis, color = 'yellow', linewidth=1)

df, loc, scale = stats.chi2.fit(data['n'])
print("chi2 fit: b={:.6f}, loc={:.6f}, scale={:.6f}".format(df, loc, scale))
chi2_pdf = stats.chi2.pdf(bin_centers,df, loc, scale)
chi2s = chi2_pdf*bin_width*len(data)
plt.plot(bin_centers, chi2s, color = 'orange', linewidth=1)
'''

'''
plt.figure(6)
fy = np.abs(np.fft.fft(data['percent_change']))
fx = np.fft.fftfreq(len(data))
plt.plot(fx, fy)
'''

data.plot(y='Close')

print('    bin actual        t   cauchy   normal')
print("\n".join("{:>+6.2f}% {:>6} {:>8.3f} {:>8.3f} {:>8.3f}"
	.format(bin_center, int(actual), t, cauchy, normal)
	for bin_center, actual, t, cauchy, normal
	in zip(bin_centers, actuals, ts, cauchies, normals)))


ngains = sum(1 for x in data[2:]['percent_change'] if x > 0)
nlosses = sum(1 for x in data[2:]['percent_change'] if x < 0)
ggain = sum(1 for i,pc in enumerate(data[2:]['percent_change']) if data.loc[i+1, 'percent_change'] > 0 and pc > 0)
gloss = sum(1 for i,pc in enumerate(data[2:]['percent_change']) if data.loc[i+1, 'percent_change'] < 0 and pc > 0)


pgain = ngains / len(data)
ploss = nlosses / len(data)

pgainggain = ggain / ngains
plossggain = 1 - pgainggain

pgaingloss = gloss / nlosses
plossgloss = 1 - pgaingloss

print(f"days{len(data)}, ngains{ngains}, nlosses{nlosses}, pgain{pgain}, ploss{ploss}")
print(f"pgainggain{pgainggain}, pgaingloss{pgaingloss}")
print(f"plossggain{plossggain}, plossgloss{plossgloss}")

pnno = nnno / len(data)
print(f"pnno{pnno}")

def play_sound(data):
	data['audio'] = data['percent_change'] / 20
	soundarray = data['audio'].values
	while True:
		fs = int(input('samplerate: '))
		sd.play(soundarray, fs)
threading.Thread(target=play_sound, args=(data,))#.start()

'''
ac = signal.correlate(data['percent_change'], data['percent_change'])
lags = signal.correlation_lags(len(data), len(data))
ac /= np.max(ac)
plt.figure(8)
plt.plot(lags, ac)
'''
'''
pc_ac, conf, qstat, pvalues = sm.tsa.stattools.acf(data['percent_change'][1:], nlags=len(data)-2, alpha=0.05, qstat=True)
plt.figure(8)
plt.plot(pc_ac[1:])
plt.plot(np.zeros(len(pc_ac)-1))
plt.plot(conf[1:])
print(qstat)
print(pvalues)
'''
import statsmodels as sm
sm.graphics.tsaplots.plot_acf(data['percent_change'], lags=len(data)-1, zero=False, alpha=0.5)


print('CLOSE:')
adf = ADF(data['Close'])
print(adf.summary().as_text())

print('PC:')
adf = ADF(data['percent_change'])
print(adf.summary().as_text())

'''
abs_ac = signal.correlate(data['abs_pc'], data['abs_pc'])
lags = signal.correlation_lags(len(data), len(data))
abs_ac /= np.max(abs_ac)
plt.figure(9)
plt.plot(lags, abs_ac)
liny = abs_ac[len(data):]
linx = lags[len(data):]
line = stats.linregress(linx, liny)
slope = line.slope
intercept = line.intercept
print(slope)
print(intercept)
print(linx[0], liny[0])
print(linx[1], liny[1])
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--')
'''
'''
abs_ac, conf, qstat, pvalues = sm.tsa.stattools.acf(data['abs_pc'], nlags=len(data)-1, alpha=0.05, qstat=True)
plt.figure(9)
plt.plot(abs_ac[1:])
plt.plot(np.zeros(len(abs_ac)-1))
plt.plot(conf[1:])
print(qstat)
print(pvalues)
'''
'''
sm.graphics.tsaplots.plot_acf(data['abs_pc'], lags=len(data)-1, zero=False)
'''
'''
#abs 1lag
plt.figure(10)
plt.scatter(data[1:-1]['abs_pc'], data[2:]['abs_pc'])
line = stats.linregress(data[1:-1]['abs_pc'], data[2:]['abs_pc'])
slope = line.slope
intercept = line.intercept
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--')
r = np.corrcoef(data[1:-1]['abs_pc'], data[2:]['abs_pc'])
print(r)

#abs 2lag
plt.figure(11)
plt.scatter(data[1:-2]['abs_pc'], data[3:]['abs_pc'])
r = np.corrcoef(data[1:-2]['abs_pc'], data[3:]['abs_pc'])
print(r)
'''

#pc 1lag
#45 degree lines on aapl
plt.figure(12)
plt.scatter(data[1:-1]['percent_change'], data[2:]['percent_change'])
line = stats.linregress(data[1:-1]['percent_change'], data[2:]['percent_change'])
slope = line.slope
intercept = line.intercept
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--')
r = np.corrcoef(data[1:-1]['percent_change'], data[2:]['percent_change'])
print(slope, intercept)

#pc 2lag
plt.figure(13)
plt.scatter(data[1:-2]['percent_change'], data[3:]['percent_change'])
r = np.corrcoef(data[1:-2]['percent_change'], data[3:]['percent_change'])

nb= 21
bw= 1
lag=1
bins=[]
dfs=[]
locs=[]
scales=[]
bms=[]
fits_fig = plt.figure()
plt.title("hists and fits given pc-{}".format(lag))
cmap = plt.cm.rainbow
xs = np.arange(-25, 25 ,0.1)
lmb = -bw*nb/2
rmb = lmb + nb*bw
norm = matplotlib.colors.Normalize(vmin=lmb, vmax=rmb)
for i in range(nb):
	bl = lmb + i*bw
	br = bl + bw
	bm = (bl+br)/2
	bms.append(bm)
	bins.append([pc for ii, pc in enumerate(data[lag+1:]['percent_change']) if data.loc[ii+1, 'percent_change'] >= bl and data.loc[ii+1, 'percent_change'] < br])
	#bins.append([pc for ii, pc in enumerate(data[2:]['percent_change']) if data.loc[ii+1, 'percent_change'] >= bl and data.loc[ii+1, 'percent_change'] < br])
	
	#if len(bins[i]) == 0:
	#	df, loc, scale = (float('nan'), float('nan'), float('nan'))
	#else:
	#	df, loc, scale = stats.t.fit(bins[i])
	#dfs.append(df)
	#locs.append(loc)
	#scales.append(scale)
	#pdf = stats.t.pdf(xs, df, loc, scale)
	if len(bins[i]) == 0:
		mu, sigma = (float('nan'), float('nan'))
	else:
		mu, sigma = stats.norm.fit(bins[i])
	pdf = stats.norm.pdf(xs, mu, sigma)
	plt.figure(fits_fig.number)
	plt.plot(xs, pdf, color=cmap(norm(bm)), linewidth=1)
	#plt.title("t fit {}% to {}%".format(bl, br))
	actuals, bin_edges = np.histogram(bins[i], bins='fd')
	bin_width = bin_edges[1] - bin_edges[0]
	plt.figure()
	plt.plot(xs, pdf, color=cmap(norm(bm)), linewidth=1)
	plt.step(bin_edges[1:]-bin_width/2, actuals/(bin_width*len(bins[i])), color=cmap(norm(bm)), alpha = 0.5)
	#print("T fit({}, {}, {}-lag): DF={:.6f}, LOC={:.6f}, SCALE={:.6f}".format(bl, br, lag, df, loc, scale))
	print("Normal fit ({}, {}, {}-lag): mu={:.6f}, sigma={:.6f}".format(bl, br, lag, mu, sigma))

plt.figure(fits_fig.number)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])  # only needed for matplotlib < 3.1
fits_fig.colorbar(sm, ax=fits_fig.axes[0])

'''
plt.figure()
plt.subplot(311).plot(bms, dfs)
plt.subplot(312).plot(bms, locs)
plt.subplot(313).plot(bms, scales)
'''


'''
#ar()
mod = AutoReg(data['percent_change'].values, 5)
#also try abs pc
res = mod.fit()
print(res.summary())
res.plot_predict()
res.plot_diagnostics()
'''
'''
def get_fit(prev_pc):
	if prev_pc < lmb:
		return dfs[0], locs[0], scales[0]
	for i in range(nb):
		bl = lmb + i*bw
		br = bl + bw
		if prev_pc >= bl and prev_pc < br:
			return dfs[i], locs[i], scales[i]
	return dfs[-1], locs[-1], scales[-1]

def next_pc(prev_pc):
	df, loc, scale = get_fit(prev_pc)
	return loc + scale*np.random.standard_t(df)

plt.ion()
plt.show()

pc_fig = plt.figure()
price_fig = plt.figure()

#markov 1 lag simulation
days = 1000
percent_changes = [0] * days
prices = [data.loc[len(data) - 1, 'Close']] * days

while True:
	percent_changes.append(next_pc(percent_changes[-1]))
	prices.append(prices[-1] * (1+percent_changes[-1]/100))
	print("${:.2f}, {:.2f}%".format(prices[-1], percent_changes[-1]))
	prices.pop(0)
	percent_changes.pop(0)
	plt.figure(price_fig.number)
	plt.clf()
	plt.plot(prices)
	plt.figure(pc_fig.number)
	plt.clf()
	plt.plot(percent_changes)
	plt.pause(0.1)
	sleep(0.01)
'''

plt.show()
