from functions.func_train import *
from stages import *
import os

task = 'wcst' #'2afc-reversal'
dt = 0.01
tau = 10*dt
N = 200
base_learning_rate = 1e-3

constants = initconstants(dt,tau,N,task)
tv        = initvariables(task,constants)
opt       = tf.optimizers.Adam(learning_rate=base_learning_rate)
activation= tf.math.tanh #tf.nn.relu

#Training with skip connections
skip = int(sys.argv[1])
seed = int(sys.argv[2])

np.random.seed(seed)
tf.random.set_seed(seed)

allcosts = []
allpens = []

for i in range(constants['num_stages']):
    if 'ratio' in stages[task]:
        ratio = stages[task]['ratio']
    else:
        ratio = 0.10*(i+1)
    iterations = constants['iterations']

    if 'blockLen_mn' in stages[task]:
        constants['blockLen_mn'] = stages[task]['blockLen_mn']
    if 'blockLen_sd' in stages[task]:
        constants['blockLen_sd'] = stages[task]['blockLen_sd']
    if 'epLen' in stages[task]:
        constants['epLen'] = stages[task]['epLen']
    if 'batchSize' in stages[task]:
        constants['batchSize'] = stages[task]['batchSize']
    if 'ruleSupWt' in stages[task]:
        constants['ruleSupWt'] = stages[task]['ruleSupWt']
    if 'thresh' in stages[task]:
        constants['thresh'] = stages[task]['thresh']
    if 'learning_rate' in stages[task]:
        opt.learning_rate.assign(stages[task]['learning_rate'])

    costs,penalties = train(constants,tv,task,activation,opt,skip,ratio,iterations)
    allcosts.append(costs)
    allpens.append(penalties)
    print('Completed ' + str(i) + ' out of ' + str(constants['num_stages']))

allcosts = np.concatenate(allcosts,axis=0)
allpens = np.concatenate(allpens,axis=0)

os.makedirs('results',exist_ok=True)
np.save('results/' + 'models_' + task + '_' + str(seed) + 'ms_' + str(N) + 'N_' + str(skip) + 'skip_costs.npy',allcosts)
np.save('results/' + 'models_' + task + '_' + str(seed) + 'ms_' + str(N) + 'N_' + str(skip) + 'skip_penalties.npy',allpens)
