from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.stats import loguniform


class loguniform_int:
	"""Integer valued version of the log-uniform distribution"""

	def __init__(self, a, b):
		self._distribution = loguniform(a, b)

	def rvs(self, *args, **kwargs):
		"""Random variable sample"""
		return self._distribution.rvs(*args, **kwargs).astype(int)


def get_data(forces_path: str = r'/Results/10_10_2023/forces.pt'):
	forces = torch.load(forces_path)
	forces = forces.view(92 * 2908, 4)
	features = forces[:, [0, 2, 3]].numpy()
	targets = forces[:, [1]].numpy().squeeze(1)
	return features, targets


def optimize_hyperparameters(data_train, target_train):
	model = Pipeline([
		("scaler", StandardScaler()),
		("regressor", HistGradientBoostingRegressor(random_state=42))
	])

	params = {
		"regressor__max_depth": loguniform_int(1, 1000),
		"regressor__l2_regularization": loguniform(1e-6, 1e3),
		"regressor__learning_rate": loguniform(0.001, 10),
		"regressor__max_leaf_nodes": loguniform_int(2, 256),
		"regressor__min_samples_leaf": loguniform_int(1, 100),
		"regressor__max_bins": loguniform_int(2, 255),
		"regressor__max_iter": loguniform_int(1, 1000),
	}

	model_random_search = RandomizedSearchCV(
		model,
		param_distributions=params,
		n_iter=100,
		cv=5,
		verbose=2,
	)
	model_random_search.fit(data_train, target_train)
	return model_random_search.best_estimator_, model_random_search.best_params_


def train_final_model(features, targets, best_params):
	model_args = {key.split('__')[1]: val for key, val in best_params.items()}
	model = Pipeline([
		("scaler", StandardScaler()),
		("regressor", HistGradientBoostingRegressor(**model_args, verbose=2))
	])
	model.fit(features, targets)
	return model


def plotter(y_test, y_pred, slice: bool):
	if slice:
		start = np.random.randint(0, max(1, len(y_test) - 100))
		end = np.random.randint(start + 100, min(len(y_pred), start + 1000) + 1)
	else:
		start, end = 0, len(y_test) - 1
	plt.figure(figsize=(14, 7))
	plt.plot(y_test[start:end], label='True Labels', color='blue')
	plt.plot(y_pred[start:end], label='Predictions', color='red', linestyle='dashed')
	plt.xlabel('Sample Index', fontsize=14)
	plt.ylabel('Value', fontsize=14)
	plt.title(f'True Labels and Predictions [{start}:{end}]', fontsize=17)
	plt.legend(loc='upper left', fontsize=14)
	plt.show()


def generate_new_data(best_model, old_forces_path=r'/Results/22_09_2023/forces.pt'):
	old_forces = torch.load(old_forces_path).numpy()
	tensors_list = []
	for i in range(len(old_forces)):
		curr_experiment = old_forces[i]
		f2 = best_model.predict(curr_experiment)[:, np.newaxis]
		f1, f3, f4 = np.hsplit(curr_experiment, 3)
		tensor = torch.from_numpy(np.concatenate((f1, f2, f3, f4), axis=1))
		tensors_list.append(tensor)
	return torch.stack(tensors_list)


if __name__ == '__main__':
	features, targets = get_data()
	X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
	best_model, best_params = optimize_hyperparameters(X_train, y_train)
	y_pred = best_model.predict(X_test)
	plotter(y_test, y_pred, True)

	# fit on all data:
	all_data_model = train_final_model(features, targets, best_params)
	new_data = generate_new_data(all_data_model)
	torch.save(new_data, r'/Results/22_09_2023/forces_f2_predicted.pt')
