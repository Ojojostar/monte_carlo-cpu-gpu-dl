import cupy
import numpy as np
import math
import time
import torch
cupy.cuda.set_allocator(None)       # no clue
from torch.utils.dlpack import from_dlpack

import numba
from numba import cuda







@cuda.jit               # defualt GPU
def monte_carlo_andtheholygrail_gpu(d_s, s_0, Ki, Ko, mu, sigma, pot,r,
                                    d_normals, snowball_path_holder, MONTHS,
                                    N_STEPS, N_PATHS, N_BATCH):
    

    # for shared memory (non)optimization
    # shared = cuda.shared.array(shape=0, dtype=numba.float32)
    # # load to shared memory
    # path_offset = cuda.blockIdx.x * cuda.blockDim.x

    # ii - overall thread index
    ii = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x

    for n in range(ii, N_PATHS * N_BATCH, stride):
        # newly added vars for N_BATCH calculations
        batch_id = n // N_PATHS
        path_id = n % N_PATHS       # equivalent to n in old code 

        snowball_path_holder[n][0] = s_0[batch_id]
        earlyexit = False
        ki = False
        mald = False
        for t in range(N_STEPS):
            # pre shared memory b_motion    
            #                                                   
            b_motion = d_normals[path_id + batch_id * N_PATHS +  t * N_PATHS * N_BATCH]

            # post shared memory b_motion
            # shared[cuda.threadIdx.x] = d_normals[path_offset + cuda.threadIdx.x + t * N_PATHS]

            dt = 1/N_STEPS
            # pre shared memory b_motion
            ds = snowball_path_holder[n][t] * mu[batch_id] * dt + snowball_path_holder[n][t] \
                                                * sigma[batch_id] * b_motion * math.sqrt(dt) 
            # post shared memory b_motion
            # ds = snowball_path_holder[n][t] * mu[batch_id] * dt + snowball_path_holder[n][t] * sigma[batch_id] * shared[cuda.threadIdx.x] * math.sqrt(dt) 
                    # no adjusting list sizes in cuda :(
            # snowball_path.append(snowball_path[t]+ds)
            snowball_path_holder[n][t+1] = snowball_path_holder[n][t] + ds
            

            # ki = snowball_path[t] + ds
            if snowball_path_holder[n][t+1] <= Ki[batch_id]:
                ki = True

            if not mald:
                for month in (0,1,2,3,4,5,6,7,8,9,10,11):                # need to do this instead because contains (in) and range are disabled
                    if t+1 == MONTHS[month]:     #startday no longer used to fake a start date in code
                        # price = t+1+startday
                        if snowball_path_holder[n][t+1] >= Ko[batch_id]:
                            price =  pot[batch_id] * t/365     # should turn t into int
                            # return snowball_path, price
                            d_s[n] =  price * math.exp(-r[batch_id] * t/N_STEPS)   # accounting for r
                            snowball_path_holder[n][-1] = d_s[n]            
                            earlyexit = True
                            mald = True
                            # print("blo got fucked\n")
                            break

        
        if not earlyexit:       # to prevent early exit getting out of bdds error
            # did not get knocked up or down
            price = pot[batch_id]
            # t  =T 
                        # CAN'T USE T CUZ CUDA IS FUCKING SHIT so use -1 instead
                        # or not ig T works now :sob:
            if ki and snowball_path_holder[n][N_STEPS] <= s_0[batch_id]:          # blo got knocked down and never recovered
                price = snowball_path_holder[n][N_STEPS] - s_0[batch_id]
            elif ki and snowball_path_holder[n][N_STEPS] <= Ko[batch_id]:          # blo got knocked down for a bit but finished above Ki
                price =0
            d_s[n] = price * math.exp(-r[batch_id])
            snowball_path_holder[n][-1] = d_s[n]    


max_len = 1000000                 # hundo thousand data points, final speed is ~40 min for 1000 data.
number_path = 500000
batch = 1
threads = 256
seed  =1999 
num = 0
max_length = max_len
N_PATHS = number_path
N_STEPS = 365
N_BATCH  =batch
# we will not be calculating a starting date since the difference is negligible and I aint rigging up
# a system to check if a certain day is a weekend or not
MONTHS = cupy.asnumpy([0, 31,59,90,120,151,181,212,243, 273,304,334])
        # SHOULD THIS BE NP ARRAY INSTEAD????
snowball_path_holder =  np.zeros(N_BATCH*N_PATHS, dtype=(np.float32,N_STEPS+1))# extra 1 is no longer for storing payoff
# self.snowball_path_holder = cupy.array(self.snowball_path_holder)
# self.T  = np.float(365.0)         # nah id lose. 
output = cupy.zeros(N_BATCH*N_PATHS, dtype = cupy.float32)
num_blocks  =(N_PATHS * N_BATCH -1) // threads +1
num_threads = threads


# Xs =  np.zeros(max_length, dtype=(np.float32,7))#
# Ys =  np.zeros(max_length, dtype=(np.float32))#          storing final data

Xss = []
Yss = []
# Xss = np.zeros((max_length, 7))
# Yss = np.zeros((max_length, 1))


# making sure self.snowball_path_holder is zeroed to avoid bug
# self.snowball_path_holder.fill(0)
s = time.time()

for i in range(max_length):
        randoms = cupy.random.normal(0,1, N_BATCH * N_PATHS * N_STEPS, dtype= cupy.float32)

        Xpre = cupy.random.rand(N_BATCH, 7, dtype = cupy.float32)
        #                        s_0,  Ki, Ko,  mu, sigma, pot, r
        Xpre = Xpre * cupy.array([4,  -2,  1,  .01,  .15,  10, .01], dtype=cupy.float32)
        X = Xpre +    cupy.array([8,   0,  0,  .02, .275,  15, .02], dtype=cupy.float32)
        # Ki and Ko will be set down here instead of the previous line to make them relative to s_0.
        X[:, 1] = X[:,0] -1         # overriding Ki and Ko 
        X[:, 2] = X[:,0] -.2        
        # print(X)
        X[:, 1] += Xpre[:,1]        # adding back the offset in Xpre after it gets overrided
        X[:, 2] += Xpre[:,2] 

        snowball_path_holder.fill(0)
                                        # d_s, s_0, Ki, Ko, mu, sigma, pot,r,
                                        # d_normals, snowball_path_holder, MONTHS,
                                        # N_STEPS, N_PATHS, N_BATCH):
        monte_carlo_andtheholygrail_gpu[(num_blocks,), (num_threads,)](
                                        output, X[:, 0], X[:, 1], X[:, 2], X[:, 3], 
                                        X[:, 4], X[:, 5], X[:, 6],
                                        randoms, snowball_path_holder, MONTHS,
                                        N_STEPS, N_PATHS, N_BATCH)
        # o = output.reshape(N_BATCH, N_PATHS)
        # Y  =o.mean(axis =1)         # getting the average of each batch
        Y = output.mean()
        X = X.mean(axis=0)
        Xss.append(X.tolist())
        Yss.append(Y.tolist())
        # Xss.append(X)
        # Yss.append(Y)
        
        # torch.save(from_dlpack(X.toDlpack()), f"snow_data/snowX_data/tensor{i}.pt")
        # torch.save(from_dlpack(Y.toDlpack()), f"snow_data/snowY_data/tensor{i}.pt")
        if(i%100000==0):
            e = time.time()
            print(i//10000, "percent of the way there! Time is now:", (e-s)/60/60, "hours")
        num+=1
        # print((from_dlpack(X.toDlpack()), from_dlpack(Y.toDlpack()))) 

v = output.mean()
cuda.synchronize()
e = time.time()
print("done!!! Yuri!!! :yuristar:")
print('time', e-s, 'v', v, 'avg time', (e-s)/500000)

# Xs = np.array(Xss)
# Ys = np.array(Yss)
Xss = np.array(Xss)
Yss = np.array(Yss)
# print(Xss)
# print(Yss)

tensorX = torch.Tensor(Xss)
tensorY = torch.Tensor(Yss)
# print(tensorX)
# print(tensorY)
torch.save(tensorX, "snow_data_tensor_train/tsnowX.pt")
torch.save(tensorY, "snow_data_tensor_train/tsnowY.pt")
# np.save("snow_npy_data/npyX", Xss)
# np.save("snow_npy_data/npyY", Yss)