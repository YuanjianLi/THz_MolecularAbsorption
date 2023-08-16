import numpy as np
import matplotlib.pyplot as plt
import argparse
import plotly.express as px
import pandas as pd


large = 22
med = 16
small = 12
params = \
	{'axes.titlesize': large,
	 'legend.fontsize': large,
	 'figure.figsize': (16, 10),
	 'axes.labelsize': large,
	 'xtick.labelsize': med,
	 'ytick.labelsize': med,
	 'figure.titlesize': large,
	 'lines.linewidth': 4.0}

plt.rcParams.update(params)
plt.style.use('seaborn-v0_8-whitegrid')


class MolecularAbsorption:
	def __init__(self, t, phi, p):
		self.T = t
		self.phi = phi
		self.p = p
		self.C = 3 * 10 ** 8
		self.mu = self.cal_mu()

	def cal_mu(self):
		p_w = 6.1121 * (1.0007 + 3.46 * 10 ** (-6) * self.p) * np.exp(17.502 * (self.T - 273.15) / (self.T - 32.18))
		return self.phi * p_w / (100 * self.p)

	def coefficient_kappa(self, f_c):
		kappa_a = (0.2205 * self.mu * (0.1303 * self.mu + 0.0294)) / \
		          ((0.4093 * self.mu + 0.0925) ** 2 + (100 * f_c / self.C - 10.835) ** 2)

		kappa_b = (2.014 * self.mu * (0.1702 * self.mu + 0.0303)) / \
		          ((0.537 * self.mu + 0.0956) ** 2 + (100 * f_c / self.C - 12.664) ** 2)

		kappa_c = 5.54 * 10 ** (-37) * f_c ** 3 - 3.94 * 10 ** (-25) * f_c ** 2 + \
		        9.06 * 10 ** (-14) * f_c - 6.36 * 10 ** (-3)

		return kappa_a + kappa_b + kappa_c

	def cal_ma_loss_db(self, f_c, distance):
		return 10 * self.coefficient_kappa(f_c) * distance * np.log10(np.e)

	def cal_ma_loss(self, f_c, distance):
		# return 10 ** ((10 * self.coefficient_kappa(f_c) * distance * np.log10(np.e)) / 10)
		return np.exp(-self.coefficient_kappa(f_c) * distance / 2)


class PlotMALoss:
	def __init__(self, list_f_c, list_distance, t, phi, p):
		self.list_f_c = list_f_c
		self.list_distance = list_distance
		self.T = t
		self.phi = phi
		self.p = p
		self.row_dis_col_f_c = np.zeros((len(self.list_distance), len(self.list_f_c)))
		self.MA = MolecularAbsorption(t=self.T, phi=self.phi, p=self.p)

	def plot(self):
		for i, d in enumerate(self.list_distance):
			for j, f in enumerate(self.list_f_c):
				self.row_dis_col_f_c[i, j] = self.MA.cal_ma_loss_db(f_c=f, distance=d)

		plt.figure(figsize=(10, 8))
		for i in range(self.row_dis_col_f_c.shape[0]):
			plt.plot(np.array(self.list_f_c) / 10 ** 9, self.row_dis_col_f_c[i, :], label=f'd={self.list_distance[i]} m')
		plt.ylabel('Molecular Absorption Loss (dB)')
		plt.xlabel('Carrier Frequency (GHz)')
		plt.legend()
		plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Plot Molecular Absorption'
	)

	parser.add_argument('-f_c', '--car_freq', default=0.3*10**12, metavar='f_c', type=float, help='the carrier frequency')
	parser.add_argument('-t', '--tem', default=296, metavar='T', type=float, help='the temperature')
	parser.add_argument('-phi', '--phi', default=0.5, metavar='phi', type=float, help='the relative humidity')
	parser.add_argument('-p', '--pressure', default=1013.25, metavar='p', type=float, help='the pressure')
	parser.add_argument('-d', '--distance', default=30, metavar='d', type=float, help='the distance')

	args = parser.parse_args()

	MA = MolecularAbsorption(t=args.tem, phi=args.phi, p=args.pressure)

	print(f'The Molecular Absorption in dB: {MA.cal_ma_loss_db(args.car_freq, args.distance)}\n\n'
	      f'The Molecular Absorption: {MA.cal_ma_loss(args.car_freq, args.distance)}\n\n'
	      f'The Molecular Absorption Coefficient: {MA.coefficient_kappa(args.car_freq)}\n\n'
	      f'The mu: {MA.mu}')

	plot = PlotMALoss(list_f_c=list(np.arange(start=280, stop=400, step=1) * 10**9),
	                  list_distance=[1, 5, 10, 50, 100, 1000], t=296, phi=.5, p=1013.25)
	plot.plot()
