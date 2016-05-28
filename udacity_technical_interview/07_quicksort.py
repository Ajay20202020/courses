
"""Implement quick sort in Python.
Input a list.
Output a sorted list.

Wiki:

algorithm quicksort(A, lo, hi) is
    if lo < hi then
        p := partition(A, lo, hi)
        quicksort(A, lo, p – 1)
        quicksort(A, p + 1, hi)
        
algorithm partition(A, lo, hi) is
    pivot := A[hi]
    i := lo        // place for swapping
    for j := lo to hi – 1 do
        if A[j] <= pivot then
            swap A[i] with A[j]
            i := i + 1
    swap A[i] with A[hi]
    return i
    

"""

def partition(A, lo, hi):
    """ Rearrange elements in place of A in the range [lo,hi] into 
        two nonempty ranges [lo:q] and [q:hi] such that each element in
        A[lo:q] is less than or equal to each element A[q:hi].
        The index q is returned by this function
    
    """
    pivot = A[hi] 
    i = lo 
    for j in range(lo, hi): # to up to hi-1
    
        # swap as long as find values in wrong side of the pivot
        if A[j] <= pivot:
            A[j], A[i] = A[i], A[j]
            i+=1
            
    # swap the pivot
    A[i], A[hi] = A[hi], A[i]
    
    return i
        

def _quicksort(A, lo, hi):
    
    if lo < hi:
        p = partition(A, lo, hi)
        _quicksort(A, lo, p-1)
        _quicksort(A, p+1, hi)

def quicksort(A):
    _quicksort(A, 0, len(A)-1)
    return A

#%% Other immplementation

def partition2(A,i,j):

    import random
    import numpy
    # partition value
    pivot = A[random.randint(i,j-1)]
    # other option try median

    # define pointers moving from start (left) and end (right)
    left = i
    right = j-1

    # iterate until left and right pointers cross
    while True:
        
        # advance left pointer until value greater or equal than pivot
        while A[left] < pivot:
            left += 1

        # advance right pointer until value less or equal than pivot
        while A[right]> pivot:
            right -= 1

        # swap if has not crossed yet
        if left < right:
            A[left], A[right] = A[right], A[left]
            left +=1
            right-=1
        
        else:
            return right + 1
                  
def _quicksort2(A, lo, hi):
    
    # sort if at least two elements
    if hi - lo > 1:
        p = partition2(A, lo, hi)
        _quicksort2(A, lo, p)
        _quicksort2(A, p, hi)

def quicksort2(A):
    _quicksort2(A, 0, len(A))
    return A

    
#%%

if __name__ == '__main__':
    test = [21, 4, 1, 3, 9, 20, 25, 6, 21, 14]
    print (quicksort(test))
    test = [21, 4, 1, 3, 9, 20, 25, 6, 21, 14]
    print (quicksort2(test))