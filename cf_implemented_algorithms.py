from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import multiprocessing
num_cores = multiprocessing.cpu_count()


def wachter_df(x, x_cf, mad=None):
	if mad is None:
		n = torch.norm(x_cf - x, p=1, dim=1)
	else:
		absv = torch.abs(x_cf - x)
		absv = torch.div(absv, mad)
		n = torch.sum(absv, dim=1)
	return n

def wachter_objective(model, x_cf, x, lmbda, target, mad=None):
	return lmbda * ((model(x_cf) - target) ** 2) + wachter_df(x, x_cf, mad)

# Sparser Wachter
def sparse_wachter_df(x, x_cf, mad=None):
	return torch.norm(x_cf - x, p=1, dim=1) + torch.norm(x_cf - x, p=2, dim=1) ** 2

def sparse_wachter_objective(model, x_cf, x, lmbda, target, mad=None):
	return lmbda * (model(x_cf) - target) ** 2 +  sparse_wachter_df(x, x_cf)


def wachter(model, data, positive_data, lmbda, target, cat_features, mad, alglr=1e-1, eps=1e-10, use_tqdm=False, sparse=False, opt='sgd', exit_on_greater=False):
	if not sparse:
		counterfactual_objective = wachter_objective
	else:
		counterfactual_objective = sparse_wachter_objective

	if mad is not None:
		mad = mad.detach().clone().cpu()

	maxes, mins = torch.max(data, dim=0)[0], torch.min(data, dim=0)[0].detach().clone().numpy()
	def get_cf(x, model, target, mad):

		repeat = True
		cur_cycle = 0

		starting_location = x.detach().clone()
		# starting_location = torch.Tensor(np.random.uniform(low=mins, high=maxes, size=(1,len(maxes))))[0]
		# starting_location = starting_location + torch.randn_like(starting_location)
		# starting_location = torch.mean(data,dim=0).detach().clone()
		lmbda = 1e-3
		found = False

		while repeat:
			x_cf = starting_location.detach().clone().unsqueeze(0)
			x_cf.requires_grad = True

			if opt == 'sgd':
				optim = torch.optim.SGD([x_cf], momentum=0.01, lr=alglr)
			elif opt == 'adam':
				optim = torch.optim.Adam([x_cf], lr=alglr)

			# update params
			itera = 0
			last_l = 0
			converged = False

			while itera < 1_000:
				# Get cf loss
				alg_l = counterfactual_objective(model, x_cf, x, lmbda, target, mad)
				optim.zero_grad()
				alg_l.backward()
				optim.step()

				# Update iteration counter
				itera += 1

				with torch.no_grad():
					x_cf[0, cat_features] = x[cat_features]

				# Check if converged
				if (torch.abs(last_l - alg_l) < eps)[0]:
					converged = True
					break

				# update convergence
				last_l = alg_l

			if (model(x_cf)[0,0] >= 0.5) or cur_cycle >= 40:
				repeat = False

			cur_cycle += 1
			lmbda *= 2

		return x_cf.detach().clone()

	# Run cf search in parallel
	if use_tqdm:
		cfs = Parallel(n_jobs=num_cores)(delayed(get_cf)(data[i].detach().clone(), model, target, mad) for i in tqdm(range(data.shape[0])))
	else:
		cfs = Parallel(n_jobs=num_cores)(delayed(get_cf)(data[i].detach().clone(), model, target, mad) for i in range(data.shape[0]))

	return torch.stack(cfs).squeeze()
