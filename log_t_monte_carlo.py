import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ^SPX
nu=2.511895
mu=0.051494
tau=0.645380
 
days = 30
trials = 100000
final_prices = []
for x in range(trials): 
	prices = [100] * days
	percent_changes = [mu+tau*np.random.standard_t(nu) for _ in range(days)] 
	for i in range(days):
		prices[i] = prices[i-1] * (1 + percent_changes[i]/100)
	final_prices.append(prices[-1])

num_bins = 100
actuals, bin_lefts, patches = plt.hist(final_prices, bins=num_bins)
xmin, xmax = plt.xlim()
bin_width = (xmax - xmin)/num_bins
bin_centers = [bin_left+bin_width/2 for bin_left in bin_lefts]

#normal fit
mu, std = stats.norm.fit(final_prices)
print("Normal fit: MU={:.6f}, STD={:.6f}".format(mu, std))
norm_pdf = stats.norm.pdf(bin_centers, mu, std)
normals = norm_pdf*bin_width*len(final_prices)
plt.plot(bin_centers, normals, color='red', linewidth=1)

#cauchy fit
loc, scale = stats.cauchy.fit(final_prices)
print("Cauchy fit: LOC={:.6f}, SCALE={:.6f}".format(loc, scale))
cauchy_pdf = stats.cauchy.pdf(bin_centers, loc, scale)
cauchies = cauchy_pdf*bin_width*len(final_prices)
plt.plot(bin_centers, cauchies, color='yellow', linewidth=1)

#t fit
df, loc, scale = stats.t.fit(final_prices)
print("T fit: DF={:.6f}, LOC={:.6f}, SCALE={:.6f}".format(df, loc, scale))
t_pdf = stats.t.pdf(bin_centers, df, loc, scale)
ts = t_pdf*bin_width*len(final_prices)
plt.plot(bin_centers, ts, color='green', linewidth=1)

print('    bin actual          t     cauchy     normal')
print("\n".join("{:>7.2f} {:>6} {:>10.3f} {:>10.3f} {:>10.3f}"
	.format(bin_center, int(actual), t, cauchy, normal)
	for bin_center, actual, t, cauchy, normal
	in zip(bin_centers, actuals, ts, cauchies, normals)))

#plt.show()
