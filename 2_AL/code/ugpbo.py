import numpy as np
import pickle
import pandas as pd
import torch
from botorch.models import FixedNoiseGP, SingleTaskGP
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from scipy.stats import norm
from botorch.acquisition.analytic import ExpectedImprovement
import matplotlib.pyplot as plt
import sys
import time

nb_runs       = 100
nb_iterations = 500

PATH = os.getcwd() # Getting current directory
descriptor_in_path = os.path.join(PATH, '../input/descriptor.csv')

df = pd.read_csv("descriptor_in_path")
df.columns

features = ['Di', 'Df', 'Dif', 'ASA(m2/gram)_1.9',
       'AV_Volume_fraction_1.9', 'AV(cm3/gram)_1.9', 'density(gram_cm3)',
       'total_degree_unsaturation', 'degree_unsaturation',
       'metallic_percentage', 'O_to_Metal_ration', 'N_to_O_ratio', 'H', 'C',
       'N', 'O', 'F', 'P', 'S', 'Cl', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'Br', 'Cd',
       'I', 'Pb']
X = df[features].values
print("shape of X: ", np.shape(X))
material_names = df['mof'].values
material_names

for i in range(np.shape(X)[1]):
    X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))
    print("feature", i, " in [", np.min(X[:, i]), ",", np.max(X[:, i]), "]")
X

target_in_path = os.path.join(PATH, '../input/C2H6-C2H4.csv')
data = pd.read_csv(target_in_path)
data.head()

y = data['C3H8/C3H6 Selectivity (1Bar)'].values
ddd = pd.DataFrame(y)
ddd.max()

with open('inputs_and_outputs.pkl', 'wb') as file:
  pickle.dump({'X': X, 'y': y, 'features': features, 'material_names': material_names, 'nb_COFs': np.size(y), 'nb_runs': nb_runs, 'nb_iterations': nb_iterations}, file)



"""# BO run"""

X = pickle.load(open('inputs_and_outputs.pkl', 'rb'))['X']
print("shape of X:", np.shape(X))

y = pickle.load(open('inputs_and_outputs.pkl', 'rb'))['y']
y = np.reshape(y, (np.size(y), 1)) # for the GP
print("shape of y:", np.shape(y))

nb_COFs = pickle.load(open('inputs_and_outputs.pkl', 'rb'))['nb_COFs']
print("# COFs:", nb_COFs)

nb_iterations = pickle.load(open('inputs_and_outputs.pkl', 'rb'))['nb_iterations']
print("# iterations:", nb_iterations)

nb_runs = pickle.load(open('inputs_and_outputs.pkl', 'rb'))['nb_runs']
print("# runs:", nb_runs)

X = torch.from_numpy(X)
y = torch.from_numpy(y)
X.size()

y.size()

X_unsqueezed = X.unsqueeze(1)

def bo_run(nb_iterations, nb_COFs_initialization, which_acquisition, verbose=False, store_explore_exploit_terms=False):
    assert nb_iterations > nb_COFs_initialization
    assert which_acquisition in ['max y_hat', 'EI', 'max sigma']
    
    # select initial COFs for training data randomly.
    # idea is to keep populating this ids_acquired and return it for analysis.
    ids_acquired = np.random.choice(np.arange((nb_COFs)), size=nb_COFs_initialization, replace=False)
    
    # keep track of exploration vs. exploitation terms ONLY for when using EI  
    if which_acquisition == "EI" and store_explore_exploit_terms:
        explore_exploit_balance = np.array([(np.NaN, np.NaN) for i in range(nb_iterations)])
    else:
        explore_exploit_balance = [] # don't bother

    # initialize acquired y, since it requires normalization
    y_acquired = y[ids_acquired]
    # standardize outputs using *only currently acquired data*
    # y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)
    
    for i in range(nb_COFs_initialization, nb_iterations):
        print("iteration:", i, end="\r")
        # construct and fit GP model
        model = SingleTaskGP(X[ids_acquired, :], y_acquired)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # set up acquisition function
        if which_acquisition == "EI":
            acquisition_function = ExpectedImprovement(model, best_f=y_acquired.max().item())
            with torch.no_grad(): # to avoid memory issues; we arent using the gradient...
                acquisition_values = acquisition_function.forward(X_unsqueezed) # runs out of memory
        elif which_acquisition == "max y_hat":
            with torch.no_grad():
                acquisition_values = model.posterior(X_unsqueezed).mean.squeeze()
        elif which_acquisition == "max sigma":
            with torch.no_grad():
                acquisition_values = model.posterior(X_unsqueezed).variance.squeeze()
        else:
            raise Exception("not a valid acquisition function")

        # select COF to acquire with maximal aquisition value, which is not in the acquired set already
        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
        for id_max_aquisition_all in ids_sorted_by_aquisition:
            if not id_max_aquisition_all.item() in ids_acquired:
                id_max_aquisition = id_max_aquisition_all.item()
                break

        # acquire this COF
        ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
        assert np.size(ids_acquired) == i + 1
        
        # if EI, compute and store explore-exploit terms that contribute to EI separately.
        if which_acquisition == "EI" and store_explore_exploit_terms:
            # explore, exploit terms of EI. requires computing EI manually, essentially. 
            y_pred = model.posterior(X_unsqueezed[id_max_aquisition]).mean.squeeze().detach().numpy()
            sigma_pred = np.sqrt(model.posterior(X_unsqueezed[id_max_aquisition]).variance.squeeze().detach().numpy())
            
            y_max = y_acquired.max().item()
            
            z = (y_pred - y_max) / sigma_pred
            explore_term = sigma_pred * norm.pdf(z)
            exploit_term = (y_pred - y_max) * norm.cdf(z)
            
            # check we computed it right... i.e. that it agrees with BO torch's EI.
            assert np.isclose(explore_term + exploit_term, acquisition_values[id_max_aquisition].item())

            explore_exploit_balance[i] = (explore_term, exploit_term)

        # update y aquired; start over to normalize properly
        y_acquired = y[ids_acquired] # start over to normalize y properly
        #y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)
        
        if verbose:
            print("\tacquired COF", id_max_aquisition, "with y = ", y[id_max_aquisition].item())
            print("\tbest y acquired:", y[ids_acquired].max().item())
        
    assert np.size(ids_acquired) == nb_iterations
    return ids_acquired, explore_exploit_balance

# which_acquisition = "EI"
# which_acquisition = "max y_hat"
which_acquisition = "max sigma"
nb_COFs_initializations = {"EI": [5, 10, 15, 20, 25], 
                           "max y_hat": [10], 
                           "max sigma": [10]}

for nb_COFs_initialization in nb_COFs_initializations[which_acquisition]:
    print("# COFs in initialization:", nb_COFs_initialization)
    # store results here.
    bo_res = dict() 
    bo_res['ids_acquired']            = []
    bo_res['explore_exploit_balance'] = []
    
    if nb_COFs_initialization == 10 and which_acquisition == 'EI':
        store_explore_exploit_terms = True
    else:
        store_explore_exploit_terms = False
    
    for r in range(nb_runs):
        print("\nRUN", r)
        t0 = time.time()
        
        ids_acquired, explore_exploit_balance = bo_run(nb_iterations, nb_COFs_initialization, which_acquisition, store_explore_exploit_terms=store_explore_exploit_terms)
        
        # store results from this run.
        bo_res['ids_acquired'].append(ids_acquired)
        bo_res['explore_exploit_balance'].append(explore_exploit_balance)
        
        print("took time t = ", (time.time() - t0) / 60, "min\n")
    
    # save results from all runs
    with open('bo_results_' + which_acquisition + "_initiate_with_{0}".format(nb_COFs_initialization) + '.pkl', 'wb') as file:
        pickle.dump(bo_res, file)
        
with open('bo_results_nb_COF_initializations.pkl', 'wb') as file:
    pickle.dump(nb_COFs_initializations, file)
