import cupy
import numpy as np
import math
import time
import torch
cupy.cuda.set_allocator(None)       # no clue
from torch.utils.dlpack import from_dlpack

import numba
from numba import cuda
import random
import os, os.path
from pathlib import Path

from datetime import datetime

# code was written with batch of size 1 in mind and number of data points gets a bit off the inputed amount
# if the division ends in a decimal

###     CHECK OUT DATASETGEN_MULTI FOR A SMALL OPTIMIZATION WHEN ACTUALLY RUNNING CODE, OR JUST RUN THIS IN MULTIPLE COMMAND PROMPTS


@cuda.jit               # defualt GPU
def monte_carlo_andtheholygrail_gpu(d_s, s_0, Ki, Ko, mu, sigma, pot,r,
                                    d_normals,day_diff,  remaining_months,start_month,
                                      snowball_path_holder, MONTHS, temp_months,
                                    N_STEPS, N_PATHS, N_BATCH, min_b):
    



    # for shared memory (non)optimization
    # shared = cuda.shared.array(shape=0, dtype=numba.float32)
    # # load to shared memory
    # path_offset = cuda.blockIdx.x * cuda.blockDim.x
    # ii - overall thread index
    ii = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x

    for n in range(ii, N_PATHS * N_BATCH, stride):
        # newly added vars for N_BATCH calculations
        batch_id = n // N_PATHS         # which batch is currently running? (only 1 exists)
        path_id = n % N_PATHS       # equivalent to n in old code 
        



        # start_month = int(start_month[0])                 # cuda breaks if u do this
        # remaining_months=  int(remaining_months[0])
        # day_diff = int(day_diff[0])

        start_month[0] -= 1          # [0] equivalent to [batch_id], just have it up here to be slightly more effecient since only running 1 batch
        
        for month in (0,1,2,3,4,5,6,7,8,9,10,11):
            temp_months[month] += int(day_diff[0])       # adds the amont of days passed to each month

        for month in range(0,int(start_month[0])):
            temp_months[month] += 365           # months adjusted for flexible start date. 
                                                # lots of math since i refuse to create extra vars for mc that
                                                # aren't featured in the NN. 

        end_month = int(start_month[0]) + int(remaining_months[0])
            # N_STEPS is how long the entire thing runs for while N_STEPS_NEW is how long the option still has left
        N_STEPS_NEW = MONTHS[end_month] - MONTHS[int(start_month[0])] + int(day_diff[0])       #
        N_STEPS_NEW = N_STEPS



        snowball_path_holder[n][0] = s_0[batch_id]
        earlyexit = False
        ki = False
        mald = False
        for t in range(N_STEPS_NEW):
            # pre shared memory b_motion    
            #                                                                           # last part added for optim
            b_motion = d_normals[path_id + batch_id * N_PATHS +  t * N_PATHS * N_BATCH + N_BATCH * N_PATHS * N_STEPS * min_b]

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
            # cuda.syncthreads()

            if not mald:
                for month in (0,1,2,3,4,5,6,7,8,9,10,11):                # need to do this instead because contains (in) and range are disabled
                    if t+1 == temp_months[month]:     #startday no longer used to fake a start date in code
                        # price = t+1+startday
                        if snowball_path_holder[n][t+1] >= Ko[batch_id]:
                            price =  pot[batch_id] * t/365     # should turn t into int
                            # return snowball_path, price
                            d_s[n] =  price * math.exp(-r[batch_id] * t/N_STEPS)   # accounting for r
                            snowball_path_holder[n][N_STEPS_NEW] = d_s[n]            
                            earlyexit = True
                            mald = True
                            # print("blo got fucked\n")
                            break
            else: # if mald
                break
            # cuda.syncthreads()

        if not earlyexit:       # to prevent early exit getting out of bdds error
            # did not get knocked up or down
            price = pot[batch_id]
            if ki and snowball_path_holder[n][N_STEPS_NEW] <= s_0[batch_id]:          # blo got knocked down and never recovered
                price = snowball_path_holder[n][N_STEPS_NEW] - s_0[batch_id]
            elif ki and snowball_path_holder[n][N_STEPS_NEW] <= Ko[batch_id]:          # blo got knocked down for a bit but finished above Ki
                price =0
            d_s[n] = price * math.exp(-r[batch_id])
            snowball_path_holder[n][N_STEPS_NEW] = d_s[n]    

        # cuda.syncthreads()
#               make sure max_len is large enough or else divide by zero error occurs (at least 100 batches must be run)
limiter = True

###              IMPORTANT VARAIBLE!!!!!
max_len = 1000000                    # hundo thousand data points, final speed is ~40 min for 1000 data.
# max_len = 500000                    # hundo thousand data points, final speed is ~40 min for 1000 data.
# max_len = 100                 
# max_len = 1000
number_path = 500000             # regular use number
number_path = 10000000            # for accurate validation data (20 times more paths)
batch = 1
splitter = 10
multiplier = 20         ## similar to splitter but generates less randoms at a single time, decreasing amount of memory used
                        # IMPORTANT!!! please shift multiplier according to paths. if path x 10, mult x10. This will keep things consistent.
                        # overall, tweaking of splitter is required to find fastest computation speed and just keep mult in line with paths.  
multiplier = number_path // 500000

    # i dont belive # of threads has like any impact on how fast it runs
# threads = 256                     # total amount of suportable threads is arond 8-900. 256 supports 3 programs runing at same time
threads = 128           
seed  =1999 
max_length = max_len
N_PATHS = (number_path // splitter) // multiplier
N_STEPS = 365
N_BATCH  =batch
max_length = max_length // N_BATCH
# max_length = max_length // 1
percenter  =100       #  changes how often percent text gets shown & how many times things are saved
percent = max_length // percenter

batch_size = 10000      # 1000000/100
batch_size = batch_size/multiplier      # using a smaller batch to compensate for x20 mult for larger path nums
# batch_size = 100      # 1000000/100               for testing smaller batches
# batch_limit = 500       # equivalent to percenter in how it limits how many files are generated
batch_limit = 500          # smaller number for training accurate validation

#           uncomment if u want less batches, the percent will just be wrong
if percent == 0:
    percent = 1


MONTHS = cupy.asnumpy([0, 31,59,90,120,151,181,212,243, 273,304,334])
MONTHS_SINGLE = cupy.asnumpy([31,28,31,30,31,30,31,31, 30,31,30,31])
snowball_path_holder =  np.zeros(N_BATCH*N_PATHS, dtype=(np.float32,N_STEPS+1))
output = cupy.zeros(N_BATCH*N_PATHS, dtype = cupy.float32)
num_blocks  =(N_PATHS * N_BATCH -1) // threads +1
num_threads = threads


Xss = []
Yss = []
Ys  =[]

path = Path(__file__).parent.absolute()
                    ###          IMPORTANT VARAIBLE!!!!!
# folder = "snow_data2"
# folder = "snow_data_tensor_test2"              
# folder = "snow_data_tensor2"         
folder = "snow_data_morepath2"
dir = f"{path}\{folder}" 

print("\nPRINGITN CURRENT DIR", dir)
currnum = len(os.listdir(dir))//2+1
print("Batch size:", N_BATCH)
print("Adding files starting from", currnum, "\n")



start = 1
if limiter:
    start = currnum         ## equivalent to saved i

s = time.time()
now = datetime.now()
print(f"current time is: {now}")

for i in range(start,max_length+1):
        # randoms = cupy.random.normal(0,1, N_BATCH * N_PATHS * N_STEPS* splitter, dtype= cupy.float32)
        snowball_path_holder.fill(0)

        Xpre = cupy.random.rand(1, 13, dtype = cupy.float32)
        #                        s_0,  Ki, Ko,  mu, sigma, pot, r, day_diff,  remaining_months, start month, start day,  end day, end month
        Xpre = Xpre * cupy.array([4,  -2,  1,  .01,  .15,  10, .01, 0, 0, 12, 1, 1, 12], dtype=cupy.float32)
        X = Xpre +    cupy.array([8,   0,  0,  .02, .275,  15, .02, 0, 0, 1, 0, 0, 1], dtype=cupy.float32)
        # Ki and Ko will be set down here instead of the previous line to make them relative to s_0.
        X[:, 1] = X[:,0] -1         # overriding Ki and Ko 
        X[:, 2] = X[:,0] -.2        
        X[:, 1] += Xpre[:,1]        # adding back the offset in Xpre after it gets overrided
        X[:, 2] += Xpre[:,2] 

        X[:, 9] = int(X[:,9]) 
        X[:, 12] = int(X[:,12]) 
        monther = MONTHS_SINGLE[int(X[:,9]) -1] 
        X[:,10] = int(X[:,10] * monther +1)
        monther = MONTHS_SINGLE[int(X[:,12]) -1] 
        X[:,11] = int(X[:,11] * monther +1)



        # generate dates        !!! im pretty sure this code breaks when batch is greater than 1. may need to for lop it.
        start_day = X[:, 10] 
        start_month = X[:, 9]         #
        end_day = X[:, 11]    
        end_month = X[:, 12]
        remaining_months = end_month-start_month        #
        if remaining_months <=0:
            remaining_months += 12
        day_diff  = end_day-start_day        # day_diff
        if start_day > end_day:         # reversed
            # print(f"remaining: {remaining_months}")
            remaining_months -=1
            # print(f"remaining: {remaining_months}")

            day_diff  = MONTHS_SINGLE[int(start_month)-1] - start_day + end_day       #
            start_month += 1
            if start_month > 12:
                # print(f"TESTING: {start_month}")
                start_month -= 12

        X[:,7] = day_diff
        X[:,8] = remaining_months
        # X[:,9] = start_month
        



        for mult in range (multiplier):
            for min_b in range(splitter):
                temp_months = cupy.asnumpy([0, 31,59,90,120,151,181,212,243, 273,304,334])
                randoms = cupy.random.normal(0,1, N_BATCH * N_PATHS * N_STEPS* splitter, dtype= cupy.float32)
                                                # d_s, s_0, Ki, Ko, mu, sigma, pot,r,
                                                # d_normals, snowball_path_holder, MONTHS,
                                                # N_STEPS, N_PATHS, N_BATCH):
                monte_carlo_andtheholygrail_gpu[(num_blocks,), (num_threads,)](
                                                output, X[:, 0], X[:, 1], X[:, 2], X[:, 3], 
                                                X[:, 4], X[:, 5], X[:, 6], randoms, 
                                                X[:, 7], X[:, 8], X[:, 9],
                                                snowball_path_holder, MONTHS,temp_months,
                                                N_STEPS, N_PATHS, N_BATCH, min_b)
                cuda.synchronize()
                

                Y = output.mean()
                Ys.append(Y.tolist())       # Ys is a list of mc things for same X and we meaning all em together

        # print (Ys)
        Xs = X[:,0:10].mean(axis=0)
        Xss.append(Xs.tolist())
        # Yss.append(Ys.mean())           # hope this works???
        Ys_avg = sum(Ys) / len(Ys)
        Yss.append(Ys_avg)

        
        # # print (Ys)
        # X = X.mean(axis=0)
        # Xss.append(torch.from_numpy(X))
        # # Yss.append(Ys.mean())           # hope this works???
        # Ys_avg = sum(Ys) / len(Ys)
        # Yss.append(torch.tensor(Ys_avg, dtype=torch.float32))


        # print(Yss)
        Ys.clear()
        # print(i)
        # torch.save(from_dlpack(X.toDlpack()), f"snow_data/snowX_data/tensor{i}.pt")
        # torch.save(from_dlpack(Y.toDlpack()), f"snow_data/snowY_data/tensor{i}.pt")

        # if(i%i==0):                   # for testing purposed only!!
        #not nesting the if statements so that first check can be easily commented out if run is very fast
        
        # if(i%(percent/10)==0):
        if(i%(batch_size/10)==0):
            e = time.time()
            currnum = len(os.listdir(dir))//2+1
            print(currnum -1 + (i/(batch_size))%1, "parts/percent of the way there! Time is now:", (e-s)/60, "minutes")
        # if(i%percent==0):
        if(i%batch_size==0):
            if limiter:
                # if currnum > percenter:
                if currnum > batch_limit:
                    print("\nPremature exit, burunyu~")
                    print(" ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢭⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                          "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣻⣶⣝⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣸⣿⣿⣿⣾⣿⣿⣿⣿⣿⡿⣫⣶⣿⣿⣿⢸⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣟⣿⣿⣿⣿⣿⣿⣮⢿⣿⣝⣿⣿⡿⢟⣛⣭⣭⣝⣿⡏⢹⣿⣿⣻⣾⣿⢟⣿⣿⣿⣿⣿⣿⣿⣼⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⡻⣼⣿⣟⣯⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣸⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣯⢿⣿⣿⡇⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣳⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣮⣿⣿⣿⣿⣿⣿⣿⡇⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣧⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⣿⣿⣿⣽⣿⣿⣿⣿⣷⠻⣿⣿⣿⣻⣾⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⣿⣿⣷⣿⣿⣿⣻⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢇⣿⣿⣿⣿⣿⣿⣿⣿⣇⣿⣿⣿⣿⣿⣿⣷⣿⣿⡼⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⢻⣿⣿⣿⣿⣿⡏⣿⣿⣿⣸⣿⣿⣿⣿⣿⣯⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⡾⣿⣿⣿⣿⣿⣿⣿⣿⣿⢹⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣏⣿⣿⣿⣿⣿⣿⢻⣿⣻⣿⢸⣿⣿⣿⣿⣻⣿⣿⣿⣿⣿⣿⣿⡟⣿⣿⣇⣿⣼⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⢸⣿⣽⣿⣿⣿⡇⣿⣛⡍⠀⣿⣿⣶⣼⢿⣿⣿⡿⣿⣻⣿⠸⣛⣿⣼⣻⣃⠀⣿⣿⣿⣿⣿⡞⣿⣿\n",
                            "⣿⣿⣿⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢳⣿⣿⣿⠀⠀⠀⣿⣿⣿⣿⣿⡽⣿⣟⣿⠀⠀⢸⣿⣿⣿⣿⣿⢻⣿⣿⣿⣼⣿⣿⣿\n",
                            "⣿⣿⣏⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⢸⣿⣿⣿⣿⣿⡇⣿⣿⣿⡇⣿⡇⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡼⣿⣿⣿⠀⠀⢠⣿⣿⣿⣿⣿⣳⣿⣻⣿⠀⠀⣿⣿⣿⣿⣿⣟⣿⣿⣿⣿⡇⣻⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⣿⣿⣿⣿⡟⣿⣿⣻⣿⣤⣿⣿⣿⢟⣿⣿⣿⣿⣿⣿⣷⣻⣿⣟⣯⣾⣿⢷⣿⣿⣿⣿⣿⣿⣿⣅\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⢿⣿⣿⣿⠀⢿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⠁⠀⣿⣿⣿⣿⢸⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣧⣿⣿⡇⠀⠀⣿⣿⡏⠀⠀⠀⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠀⠀⢸⣿⣝⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣤⣿⣿⣿⣿⡆⠀⠀⢰⡀⠀⠀⠙⠻⠿⣿⣿⣿⡿⠟⠋⠁⠀⠀⣿⠀⠀⠀⣿⣿⣿⣿⣷⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣾⣿⣿⣴⣿⣿⣿⣿⣿⣿⣿⣺⣿⣮⣦⠀⢸⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⡿⣵⣿⣮⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣳⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣞⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⢹⣿⣿⣿⣿⣼⣿⣿⣿⣿⣿⣿⣿⣿⣳⣿⣿⣿⣿⣿⣧⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⢸⣿⣿⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢻⣝⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢘⣶⣶⣯⢿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣻⣿⣿⣿⣿⣝⣿⣿⣿⣿⣿⣿⣿⣿⣿⣻⣿⣦⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢇⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣯⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣯⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠉⠉⠛⠛⠛⠉⠉⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⡿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣝⠻⠿⣛⣾⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣿⣿⠟⣶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⠀⠀⠀⠀⠀⠀⣶⣄⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⣼⣿⣿⣿⣿⣿⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⣼⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣷⣄⣠⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣀⠀⣠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n",
                            "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n")
                    # Xss.pop()
                    # Yss.pop()
                    break
            
            tensorX = np.array(Xss)
            tensorY = np.array(Yss)
            tensorX = torch.Tensor(tensorX)
            tensorY = torch.Tensor(tensorY)
            e = time.time()
            currnum = len(os.listdir(dir))//2+1         #equivalent to curnum+=1 but allows for paralelel running
            # print(i/(percent), "percent of the way there! Time is now:", e-s, "secs")
            print(currnum, "parts/percent of the way there! Time is now:", (e-s)/60/60, "hours")
            # print(currnum, "percent of the way there! Time is now:", (e-s)/60, "mins")
            print(f"current time is: {now}")
            print("now saving tsnowX_{}.pt\n".format(currnum) )
            torch.save(tensorX, f"{dir}\\tsnowX_{currnum}.pt")
            torch.save(tensorY, f"{dir}\\tsnowY_{currnum}.pt")
            Xss.clear()
            Yss.clear()
            # currnum += 1

v = output.mean()
cuda.synchronize()
e = time.time()
print("done!!! Yuri!!! :yuristar:")
print('time', (e-s)/60/60, 'hours')


#           may need to use if percent doesn't end on 100
# if Xss:                     # chekcing if its empty of not. only pritn if isnt.
#     print("this seems odd...", len(Xss))
#     Xss = np.array(Xss)
#     Yss = np.array(Yss)
#     tensorX = torch.Tensor(Xss)
#     tensorY = torch.Tensor(Yss)
#     torch.save(tensorX, "snow_data_tensor_train/tsnowX.pt")
#     torch.save(tensorY, "snow_data_tensor_train/tsnowY.pt")