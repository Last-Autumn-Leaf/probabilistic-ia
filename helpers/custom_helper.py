
from collections import defaultdict
def getHighestFrequencyVector(image) :
    store=defaultdict(int)
    x,y,z=image.shape
    for i in range(x) :
        for j in range(y):
            store[tuple(image[i,j])]+=1

    max_bin=0

    for k,v in store.items():
        if max_bin< v :
            max_bin=v
            max_vector=k
    return max_bin,max_vector