import numpy as np
import os
import time
#import matplotlib.pyplot as plt
import tensorflow as tf
import sys

tf.config.set_visible_devices([], 'GPU')
# gpus_all_physical_list = tf.config.list_physical_devices(device_type='GPU')
# print(gpus_all_physical_list)
# tf.config.set_visible_devices(gpus_all_physical_list[0], 'GPU')
    
def initconstants(dt,tau,N,task):

    constants = {}

    constants['tau_e']      = tf.constant(tau, dtype=tf.float32)
    constants['tau_i']      = tf.constant(tau/2, dtype=tf.float32)
    constants['tau_n']      = tf.constant(tau, dtype=tf.float32)
    constants['dt']         = tf.constant(dt, dtype=tf.float32)

    constants['eps1']       = tf.constant((1.0-dt/tau), dtype=tf.float32)
    constants['eps2']       = tf.constant(np.sqrt(2.0*dt/tau), dtype=tf.float32)

    constants['N_exc']      = int(0.8*N)
    constants['N_inh']      = int(0.2*N)
    constants['N']          = N

    Tinvdiag                = np.ones([N])/tau
    # Tinvdiag[int(0.8*N):]   = 2/tau
    constants['Tinv']       = tf.constant(np.diag(Tinvdiag), dtype=tf.float32)

    mask                    = np.ones([N,N])
    mask[:,int(0.8*N):]     = -1
    constants['mask']       = tf.constant(mask, dtype=tf.float32)

    if task == '2afc-reversal':

        constants['N_out']  = 2
        constants['N_stim'] = 8 # 2 for left stim, 2 for right stim, 2 for action, 2 for reward
        constants['T_stim'] = int(0.5/dt)
        constants['T_wait'] = int(0.5/dt)
        constants['T_resp']  = int(0.5/dt)
        constants['T_FB']  = int(0.1/dt)
        constants['T_ITI']  = int(0.5/dt)
        constants['blockLen_mn']  = 15. # mean no. of trials with given rule
        constants['blockLen_sd']  = 3. # std of no. of trials with given rule
        constants['epLen']  = 50 # no. of consecutive trials simulated and trained on
        constants['batchSize'] = 20
        constants['iterations'] = 2000
        constants['num_stages'] = 10

    elif task == 'wcst':
        constants['N_out'] = 7
        constants['N_stim_singleCard'] = 12
        constants['N_stim'] = constants['N_stim_singleCard']*2 + 6 # 12 for single reference card, 12 for test card, 4 for action, 2 for reward
        constants['T_stim_singleCard'] = int(0.25/dt)
        constants['T_stim'] = constants['T_stim_singleCard']*4
        constants['T_resp']  = int(0.5/dt)
        constants['T_FB']  = int(0.5/dt)
        constants['T_ITI']  = int(0.5/dt)
        constants['blockLen_mn']  = 8. #15. # mean no. of trials with given rule
        constants['blockLen_sd']  = 2. #3. # std of no. of trials with given rule
        constants['epLen']  = 15 #50 # no. of consecutive trials simulated and trained on
        constants['batchSize'] = 100
        constants['ruleSupWt'] = 1.0
        constants['thresh'] = 2
        constants['iterations'] = 30000
        constants['num_stages'] = 13

    return constants

def initvariables(task,constants):
    initializer = tf.keras.initializers.Orthogonal(gain=1)#0.9)
    Wraw = initializer(shape=[constants['N'], constants['N']])
    Wraw = tf.Variable(Wraw, dtype=tf.float32, name='Wraw') #Raw weight matrix before Dale's Law and other constraints
    bias = tf.Variable(tf.random.normal(shape=[constants['N'],1]), dtype=tf.float32, name='bias') #Input bias

    if task == '2afc-reversal' or task == 'wcst':

        Win  = tf.Variable(tf.random.normal(shape=[constants['N'],constants['N_stim']])/(constants['N']), dtype=tf.float32, name='Win')
        Wout = tf.Variable(tf.random.normal(shape=[constants['N_out'], constants['N']])/(constants['N']), dtype=tf.float32, name='Wout')

        tv = [Wraw,bias,Win,Wout]

        return tv
