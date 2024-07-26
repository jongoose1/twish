import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

ticker = input("ticker: ").upper()
try:
	data = pd.read_pickle(ticker+'.pkl')
	print("Previous data found")
except:
	data = yf.download(ticker, period='max')
	if data.empty:
		exit()
	data.to_pickle(ticker+'.pkl')

data = data.reset_index()
data.loc[0, 'percent_change'] = 0
for i in range(1, len(data)):
	data.loc[i, 'percent_change'] = 100* (data.loc[i, 'Close'] - data.loc[i-1, 'Close']) / data.loc[i-1, 'Close']

data.plot(y = 'percent_change')
plt.figure(2)
num_bins = 150
actuals, bin_lefts, patches = plt.hist(data['percent_change'], bins=num_bins)
xmin, xmax = plt.xlim()
#x = np.linspace(xmin, xmax, num_bins)
bin_width = (xmax-xmin)/num_bins
bin_centers = [bin_left+bin_width/2 for bin_left in bin_lefts]

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

data.plot(y='Close')

print('    bin actual        t   cauchy   normal')
print("\n".join("{:>+6.2f}% {:>6} {:>8.3f} {:>8.3f} {:>8.3f}"
	.format(bin_center, int(actual), t, cauchy, normal)
	for bin_center, actual, t, cauchy, normal
	in zip(bin_centers, actuals, ts, cauchies, normals)))

#plt.show()

