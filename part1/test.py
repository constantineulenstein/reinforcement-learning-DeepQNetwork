import numpy as np

all_states = np.zeros([100,2])
i = 0
for col in range(10):
    for row in range(10):
        all_states[i,0] = col/10 + 0.05
        all_states[i,1] = row/10 + 0.05
        i+=1
print(all_states)
