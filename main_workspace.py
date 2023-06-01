from functions.func_train import *
from stages import *
import os

task = 'wcst' #'wcst' #'2afc-reversal'
dt = 0.01
tau = 10*dt
N = 200
base_learning_rate = 1e-3

constants = initconstants(dt,tau,N,task)
tv        = initvariables(task,constants)
opt       = tf.optimizers.Adam(learning_rate=base_learning_rate)
activation= tf.math.tanh #tf.nn.relu

#Training with skip connections
skip = int(sys.argv[1]) # skip length in time-steps
seed = int(sys.argv[2])

np.random.seed(seed)
tf.random.set_seed(seed)

allcosts = []
allpens = []

# train network as per task-specific curriculum stages
for i in range(constants['num_stages']):
    if 'ratio' in stages[task][i]: # skip weighting factor (for annealing)
        ratio = stages[task][i]['ratio']
    else:
        ratio = 0.10*(i+1)
    iterations = constants['iterations']

    if 'blockLen_mn' in stages[task][i]:
        constants['blockLen_mn'] = stages[task][i]['blockLen_mn']
    if 'blockLen_sd' in stages[task][i]:
        constants['blockLen_sd'] = stages[task][i]['blockLen_sd']
    if 'epLen' in stages[task][i]:
        constants['epLen'] = stages[task][i]['epLen']
    if 'batchSize' in stages[task][i]:
        constants['batchSize'] = stages[task][i]['batchSize']
    if 'ruleSupWt' in stages[task][i]:
        constants['ruleSupWt'] = stages[task][i]['ruleSupWt']
    if 'thresh' in stages[task][i]:
        constants['thresh'] = stages[task][i]['thresh']
    if 'learning_rate' in stages[task][i]:
        opt.learning_rate.assign(stages[task][i]['learning_rate'])

    costs,penalties = train(constants,tv,task,activation,opt,skip,ratio,iterations)
    allcosts.append(costs)
    allpens.append(penalties)
    print('Completed ' + str(i) + ' out of ' + str(constants['num_stages']) + ' stages.')

allcosts = np.concatenate(allcosts,axis=0)
allpens = np.concatenate(allpens,axis=0)

os.makedirs('results',exist_ok=True)
np.save('results/' + 'models_' + task + '_' + str(seed) + 'ms_' + str(N) + 'N_' + str(skip) + 'skip_costs.npy',allcosts)
np.save('results/' + 'models_' + task + '_' + str(seed) + 'ms_' + str(N) + 'N_' + str(skip) + 'skip_penalties.npy',allpens)
