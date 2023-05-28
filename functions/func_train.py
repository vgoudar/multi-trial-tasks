import numpy as np

from functions.func_init import *

# -------------------------------------------------------
# Simulate dynamics
# -------------------------------------------------------

@tf.function
def compute_dueta(u,eta,W,bias,hinput,constants,activation):

    curr = W@u + bias + 0.1*eta + hinput
    du  = constants['Tinv']@(-u + tf.nn.relu(100.0*activation(curr/100.0)))*constants['dt']
    eta = constants['eps1']*eta + constants['eps2']*tf.random.normal(tf.shape(u))

    return du,eta

@tf.function
def compute_uall(u,eta,W,bias,hinput,constants,skip,ratio,timesteps,activation, t_elapsed, uskip):

    if skip>0 and t_elapsed == 0:
        du,eta = compute_dueta(u,eta,W,bias,hinput,constants,activation)
        uskip = tf.identity(u) + skip*du
    uall  = tf.TensorArray(dtype=tf.float32, size=timesteps)

    for t in tf.range(timesteps):

        du,eta = compute_dueta(u,eta,W,bias,hinput,constants,activation)
        u = u + du

        if skip>0 and (t_elapsed+t)%skip==(skip-1):
            u = ratio*u + (1-ratio)*uskip
            uskip = tf.identity(u) + skip*du
        
        uall = uall.write(t,u)

    return u,eta, uskip, uall.stack()

# -------------------------------------------------------
# Task-specific training functions
# FIXED GRAPHS only (i.e. accelerated by @tf.function)
# -------------------------------------------------------
@tf.function
def train_2afcrev(constants, tv, activation, opt, skip, ratio, input0, input1, input2, input2Var, target, ruleState, fbs, lossMask):
    activitypenalty = 1e-4
    Wraw = tv[0]
    bias = tv[1]
    Win = tv[2]
    Wout = tv[3]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    upenalty = 0.
    u = tf.zeros([constants['batchSize'] , constants['N'], 1])
    eta = tf.random.normal(tf.shape(u))

    # W = tf.abs(Wraw) * constants['mask']
    W = Wraw
    input0_curr = Win @ input0
    # u, eta, uskip, _ = compute_uall(u, eta, W, bias, input0_curr, constants, skip, ratio, 5 * constants['T_stim'], activation, 0, None)
    # t_elapsed = 5 * constants['T_stim']
    u, eta, uskip, _ = compute_uall(u, eta, W, bias, input0_curr, constants, skip, ratio, 1, activation, 0, None)
    t_elapsed = tf.constant(1)

    epLen = constants['epLen']
    cost = 0.
    epsilon = 0.0000000001
    with tf.GradientTape() as tape:
        for i in tf.range(epLen):
            perm = tf.constant(np.random.permutation(constants['batchSize']), dtype=tf.int32)
            input1 = tf.gather(input1, perm, axis=0)
            target = tf.gather(target, perm, axis=0)
            input1_curr = Win @ input1
            #W = tf.abs(Wraw) * constants['mask']
            W = Wraw

            # Sample stimulus
            timesteps = constants['T_stim']
            u, eta, uskip, uall = compute_uall(u, eta, W, bias, input1_curr, constants, skip, ratio, timesteps, activation, t_elapsed, uskip)
            upenalty += activitypenalty * tf.reduce_mean(uall ** 2)
            t_elapsed += timesteps

            # Delay period
            timesteps = constants['T_wait']
            u, eta, uskip, uall = compute_uall(u, eta, W, bias, input0_curr, constants, skip, ratio, timesteps, activation, t_elapsed, uskip)
            upenalty += activitypenalty * tf.reduce_mean(uall ** 2)
            t_elapsed += timesteps

            # Response period
            timesteps = constants['T_resp']
            u, eta, uskip, uall = compute_uall(u, eta, W, bias, input0_curr, constants, skip, ratio, timesteps, activation, t_elapsed, uskip)
            upenalty += activitypenalty * tf.reduce_mean(uall ** 2)
            t_elapsed += timesteps

            # Calculate loss
            timebroadcast = tf.ones([timesteps, 1])
            logits = (Wout @ uall)[:, :, :, 0]
            pred = tf.nn.softmax(logits, axis=2)
            pred = tf.where(pred == 1, pred-epsilon, pred)
            pred = tf.where(pred == 0, pred+epsilon, pred)
            
            targ = (1.-ruleState[:,i])*target + ruleState[:,i]*(1.-target)

            X = tf.reduce_mean(loss(targ * timebroadcast, pred), axis=0)
            X = tf.reduce_mean(X*lossMask[:,i])
            cost = cost + X

            # Generate feedback
            with tape.stop_recording():
                action = tf.cast(tf.math.argmax(pred[-1,:,:], axis=1), tf.float32)
                fb = tf.cast(targ == action, tf.float32)
                fbs[i,:].assign(fb)
                input2Var.assign(input2)
                input2Var[:,4,0].assign(action)
                input2Var[:,5,0].assign(1-action)
                input2Var[:,6,0].assign(fb)
                input2Var[:,7,0].assign(1-fb)
            input2_curr = Win @ input2Var

            # Feedback period
            timesteps = constants['T_FB']
            u, eta, uskip, uall = compute_uall(u, eta, W, bias, input2_curr, constants, skip, ratio, timesteps, activation, t_elapsed, uskip)
            upenalty += activitypenalty * tf.reduce_mean(uall ** 2)
            t_elapsed += timesteps

            # ITI period
            timesteps = constants['T_ITI']
            u, eta, uskip, uall = compute_uall(u, eta, W, bias, input0_curr, constants, skip, ratio, timesteps, activation, t_elapsed, uskip)
            upenalty += activitypenalty * tf.reduce_mean(uall ** 2)
            t_elapsed += timesteps

        wreg = 0.0001*tf.math.reduce_sum(tf.math.square(W))

        # loss = tf.reduce_mean(tf.where(tf.math.is_nan(upenalty), wreg, cost + upenalty))
        loss = cost + upenalty + wreg
        grads = tape.gradient(loss, tv)

    #grads = [tf.clip_by_norm(g, 2) for g in grads]            
    opt.apply_gradients(zip(grads, tv))

    return cost, upenalty, wreg

@tf.function
def train_wcst(constants, tv, activation, opt, skip, ratio, input0, input1, input2, input3, input4, input4Var, target, fbs,lossMask, ruleState, t0, tVar):
    activitypenalty = 1e-4
    Wraw = tv[0]
    bias = tv[1]
    Win = tv[2]
    Wout = tv[3]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss2 = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

    upenalty = 0.
    u = tf.zeros([constants['batchSize'] , constants['N'], 1])
    eta = tf.random.normal(tf.shape(u))

    #W = tf.abs(Wraw) * constants['mask']
    W = Wraw
    input0_curr = Win @ input0
    # u, eta, uskip, _ = compute_uall(u, eta, W, bias, input0_curr, constants, skip, ratio, 5 * constants['T_stim']//2, activation, 0, None)
    # t_elapsed = 5 * constants['T_stim']//2
    u, eta, uskip, _ = compute_uall(u, eta, W, bias, input0_curr, constants, skip, ratio, 1, activation, 0, None)
    t_elapsed = tf.constant(1)

    epLen = constants['epLen']
    cost = 0.
    cost1 = 0.
    cost2 = 0.
    epsilon = 0.0000000001
    with tf.GradientTape() as tape:
        for i in tf.range(epLen):

            # Sample epoch
            for j in range(4): # loop through target cards
                input = tf.concat([input1[:,:,j,:], input2[:,:,i,:], input3], axis = 1)
                input_curr = Win @ input
                #W = tf.abs(Wraw) * constants['mask']
                W = Wraw

                # target card + test card
                timesteps = constants['T_stim_singleCard']
                u, eta, uskip, uall = compute_uall(u, eta, W, bias, input_curr, constants, skip, ratio, timesteps, activation, t_elapsed, uskip)
                upenalty += activitypenalty * tf.reduce_mean(uall ** 2)
                t_elapsed += timesteps

            # Response period
            timesteps = constants['T_resp']
            u, eta, uskip, uall = compute_uall(u, eta, W, bias, input0_curr, constants, skip, ratio, timesteps, activation, t_elapsed, uskip)
            upenalty += activitypenalty * tf.reduce_mean(uall ** 2)
            t_elapsed += timesteps

            # Calculate loss
            timebroadcast = tf.ones([timesteps, 1])

            logits = (Wout @ uall)[:, :, :, 0]
            pred = tf.nn.softmax(logits[:,:,:4], axis=2)
            pred = tf.where(pred == 1, pred - epsilon, pred)
            pred = tf.where(pred == 0, pred + epsilon, pred)

            targ = target[:,i]

            X = tf.reduce_mean(loss(targ * timebroadcast, pred), axis=0)
            X = tf.reduce_mean(X * lossMask[:, i])
            cost1 = cost1 + X
            cost = cost + X

            # Generate feedback
            with tape.stop_recording():
                # action0 = tf.math.argmax(pred[-1,:,:], axis=1)
                # action0 = tf.math.argmax(tf.reduce_mean(pred, axis=0), axis=1)
                action0 = tf.squeeze(tf.random.categorical(tf.math.log(tf.reduce_mean(pred, axis=0)),1))
                action = tf.cast(action0, tf.float32)
                fb = tf.cast(targ == action, tf.float32)
                fbs[i,:].assign(fb)
                input4Var.assign(input4)

                actionOneHot =  tf.one_hot(action0, 4, dtype = tf.float32)
                input4Var[:,24:28,0].assign(actionOneHot)
                input4Var[:,28,0].assign(fb)
                input4Var[:,29,0].assign(1-fb)
            input4_curr = Win @ input4Var

            # Feedback period
            timesteps = constants['T_FB']
            u, eta, uskip, uall = compute_uall(u, eta, W, bias, input4_curr, constants, skip, ratio, timesteps, activation, t_elapsed, uskip)
            upenalty += activitypenalty * tf.reduce_mean(uall ** 2)
            t_elapsed += timesteps

            # Auxiliary target - hypothesized rules
            X = tf.expand_dims(action0, 1)
            X = tf.expand_dims(tf.tile(X, [1, constants['N_stim_singleCard']]), 2)
            Y = tf.gather_nd(input1, X, batch_dims=2)

            if i > 0:
                X = ruleState[:, i - 1] == ruleState[:, i]
                X = tf.tile(tf.expand_dims(X, axis=1), tf.constant([1,3], tf.int32))
                A = tf.where(X, tVar, t0)
            else:
                A = tVar

            for j in tf.range(3):
                X = tf.reduce_sum(tf.multiply(input2[:, (j*4):((j+1)*4), i, 0], Y[:, (j*4):((j+1)*4), 0]), axis=1)
                tVar[:, j].assign(tf.multiply(A[:, j], (tf.multiply(1.0 - X, 1.0 - fb) + tf.multiply(X, fb))))

            tVar_p = tf.nn.softmax(tVar*1000, axis=1)

            # ITI period
            timesteps = constants['T_ITI']
            u, eta, uskip, uall = compute_uall(u, eta, W, bias, input0_curr, constants, skip, ratio, timesteps, activation, t_elapsed, uskip)
            upenalty += activitypenalty * tf.reduce_mean(uall ** 2)
            t_elapsed += timesteps

            # Auxiliary loss
            logits = (Wout @ uall)[:, :, :, 0]
            pred2 = tf.nn.softmax(logits[:, :, 4:], axis=2)
            pred2 = tf.where(pred2 == 1, pred2 - epsilon, pred2)
            pred2 = tf.where(pred2 == 0, pred2 + epsilon, pred2)

            timebroadcast = tf.ones([timesteps, 1, 1])
            X = tf.reduce_mean(loss2(tVar_p * timebroadcast, pred2), axis=0)
            
            X = tf.reduce_mean(X)
            cost2 = cost2 + X
            cost = cost + constants['ruleSupWt']*X

        wreg = 0.0001 * tf.math.reduce_sum(tf.math.square(W))

        loss =  cost+upenalty+wreg
        grads = tape.gradient(loss, tv)

    #grads = [tf.clip_by_norm(g, 2) for g in grads]
    opt.apply_gradients(zip(grads, tv))

    return cost1, cost2, upenalty, wreg


# -------------------------------------------------------
# Final training function
# NO TRACING HERE
# -------------------------------------------------------
def train(constants,tv,task,activation,opt,skip,ratio,iterations):

    if task == '2afc-reversal':
        input0 = np.zeros([constants['batchSize'], constants['N_stim'], 1])
        input1 = np.zeros([constants['batchSize'], constants['N_stim'], 1])
        input2 = np.zeros([constants['batchSize'], constants['N_stim'], 1])
        target = np.zeros([constants['batchSize']])
        for i in range(constants['N_stim']):
            input1[0:(constants['batchSize']//2), 0] = 1
            input1[0:(constants['batchSize']//2), 3] = 1
            # target[0:(constants['batchSize']//2),:] = 0
            input1[(constants['batchSize']//2):, 1] = 1
            input1[(constants['batchSize']//2):, 2] = 1
            target[(constants['batchSize']//2):] = 1

        input0 = tf.constant(input0, dtype=tf.float32)
        input1 = tf.constant(input1, dtype=tf.float32)
        input2 = tf.constant(input2, dtype=tf.float32)
        target = tf.constant(target, dtype=tf.float32)

        costs = []
        penalties = []

        for iter in tf.range(iterations):
            ruleState = np.zeros((constants['batchSize'], constants['epLen']), dtype=int)-1
            lossMask = np.zeros((constants['batchSize'], constants['epLen']))+1.
            currRule = np.random.randint(2, size=constants['batchSize'])
            blockLength = constants['blockLen_mn'] + constants['blockLen_sd'] * np.random.randn(constants['batchSize'])
            blockLength = blockLength.astype(int)
            inds = np.arange(constants['batchSize'])
            prev = np.zeros((constants['batchSize']),dtype=int)
            while True:
                for i,j in enumerate(inds):
                    ruleState[j,prev[i]:(prev[i]+blockLength[i])] = currRule[i]
                    lossMask[j, prev[i]] = 0
                prev = prev+blockLength
                currRule = np.delete(currRule, np.nonzero(prev>=constants['epLen']))
                inds = np.delete(inds, np.nonzero(prev>=constants['epLen']))
                prev = np.delete(prev, np.nonzero(prev>=constants['epLen']))
                if prev.size == 0:
                    break
                blockLength = constants['blockLen_mn'] + constants['blockLen_sd'] * np.random.randn(
                    prev.shape[0])
                blockLength = blockLength.astype(int)
                blockLength = np.where((prev+blockLength) > constants['epLen'], constants['epLen']-prev, blockLength)
                currRule = 1 - currRule

            ruleState = tf.constant(ruleState, dtype=tf.float32)
            lossMask = tf.constant(lossMask, dtype=tf.float32)
            input2Var = tf.Variable(initial_value=input2, trainable=False, dtype=tf.float32)
            fbs = tf.Variable(tf.zeros(shape=(ruleState.shape[1], ruleState.shape[0])), dtype=tf.float32,
                              trainable=False)

            cost, penalty, wreg = train_2afcrev(constants, tv, activation, opt, skip, ratio, input0, input1, input2, input2Var, target,
                                                ruleState, fbs, lossMask)
            costs.append(cost.numpy())
            penalties.append(penalty.numpy())

            if iter % 10 == 0:
                ruleState = ruleState.numpy()
                ruleState = np.mean(np.sum(ruleState[:,:-1] != ruleState[:,1:], axis=1))
                fbs = np.mean(np.sum(fbs.numpy(), axis=0))

                tf.print(skip, iter, cost, penalty, wreg)
                print([ruleState, fbs], flush=True)

    elif task == 'wcst':
        input0 = np.zeros([constants['batchSize'], constants['N_stim'], 1])
        input1_np = np.zeros([constants['batchSize'], constants['N_stim_singleCard'], 4, 1])
        input3 = np.zeros([constants['batchSize'], constants['N_stim']-2*constants['N_stim_singleCard'], 1])
        input4 = np.zeros([constants['batchSize'], constants['N_stim'], 1])

        input1_np[:, [0,4,8], 0] = 1
        input1_np[:, [1,5,9], 1] = 1
        input1_np[:, [2,6,10], 2] = 1
        input1_np[:, [3,7,11], 3] = 1

        input0 = tf.constant(input0, dtype=tf.float32)
        input1 = tf.constant(input1_np, dtype=tf.float32)
        input3 = tf.constant(input3, dtype=tf.float32)
        input4 = tf.constant(input4, dtype=tf.float32)

        costs = []
        penalties = []
        rss = []
        fbss = []

        for iter in tf.range(iterations):
            input2 = np.zeros([constants['batchSize'], constants['N_stim_singleCard'], constants['epLen'], 1])
            target = np.zeros([constants['batchSize'], constants['epLen']])

            ruleState = np.zeros((constants['batchSize'], constants['epLen']), dtype=int)-1
            blockLength = constants['blockLen_mn'] + constants['blockLen_sd'] * np.random.randn(constants['batchSize'])
            blockLength = blockLength.astype(int)
            inds = np.arange(constants['batchSize'])
            prev = np.zeros((constants['batchSize']),dtype=int)
            lossMask = np.zeros((constants['batchSize'], constants['epLen'])) + 1.
            while True:
                for i,j in enumerate(inds):
                    if prev[i] == 0:
                        currRule = np.random.randint(3)
                    else:
                        currRule = np.random.randint(2)
                        if ruleState[j,prev[i]-1] == 0 or (ruleState[j,prev[i]-1] == 1 and currRule == 1):
                            currRule += 1
                    ruleState[j,prev[i]:(prev[i]+blockLength[i])] = currRule
                    lossMaskEnd = prev[i]+2
                    if lossMaskEnd > constants['epLen']:
                        lossMaskEnd = constants['epLen']
                    lossMask[j, prev[i]:lossMaskEnd] = 0
                prev = prev+blockLength
                inds = np.delete(inds, np.nonzero(prev>=constants['epLen']))
                prev = np.delete(prev, np.nonzero(prev>=constants['epLen']))
                if prev.size == 0:
                    break
                blockLength = constants['blockLen_mn'] + constants['blockLen_sd'] * np.random.randn(
                    prev.shape[0])
                blockLength = blockLength.astype(int)
                blockLength = np.where((prev+blockLength) > constants['epLen'], constants['epLen']-prev, blockLength)

            input4Var = tf.Variable(initial_value=input4, trainable=False, dtype=tf.float32)
            fbs = tf.Variable(tf.zeros(shape=(ruleState.shape[1], ruleState.shape[0])), dtype=tf.float32,
                              trainable=False)

            i1 = np.arange(constants['batchSize'])
            for i in range(constants['epLen']):
                for k in range(3):
                    d = np.random.randint(4, size=(constants['batchSize'])) + k*4
                    input2[i1, d, i, 0] = 1

                for j in range(4):
                    for k in range(3):
                        target[:, i] += j * np.einsum('ik,ik->i', input2[:, (k*4):((k+1)*4), i, 0], input1_np[:, (k*4):((k+1)*4), j, 0]) * \
                                        (ruleState[:, i] == k).astype(np.float32)

            input2 = tf.constant(input2, dtype=tf.float32)
            target = tf.constant(target, dtype=tf.float32)
            lossMask = tf.constant(lossMask, dtype=tf.float32)
            ruleState = tf.constant(ruleState, dtype=tf.int32)
            t0 = tf.constant(np.ones((constants['batchSize'], 3)), dtype=tf.float32)
            tVar = tf.Variable(initial_value=t0, trainable=False, dtype=tf.float32)

            cost1, cost2, penalty, wreg = train_wcst(constants, tv, activation, opt, skip, ratio, input0, input1, input2, input3, input4, input4Var, target,
                                                     fbs, lossMask, ruleState, t0, tVar)
            costs.append(cost1.numpy()+cost2.numpy())
            penalties.append(penalty.numpy())

            ruleState = np.mean(np.sum(ruleState[:,:-1] != ruleState[:,1:], axis=1))
            fbs = np.mean(np.sum(fbs.numpy(), axis=0))
            
            rss.append(ruleState)
            fbss.append(fbs)
            if len(rss) > 100:
                if np.sum(np.array(fbss[-100:]) >  (constants['epLen'] - constants['thresh']*(np.array(rss[-100:])+1)  )) == 100:
                    return costs, penalties
            
            if iter % 10 == 0:
                tf.print(skip, iter, cost1, cost2, penalty, wreg)
                print([ruleState, fbs], flush=True)

    return costs, penalties

