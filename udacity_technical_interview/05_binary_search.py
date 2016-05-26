"""You're going to write a binary search function.
You should use an iterative approach - meaning
using loops.
Your function should take two inputs:
a Python list to search through, and the value
you're searching for.
Assume the list only has distinct elements,
meaning there are no repeated values, and 
elements are in a strictly increasing order.
Return the index of value, or -1 if the value
doesn't exist in the list."""

def binary_search(input_array, value):
    """Your code goes here."""
    
    N = len(input_array)
    
    # indices/pointers
    left = 0
    right = N-1
    middle = (right - left)//2 + left # size is odd (3) = middle (1), size is even (4), lower middle (1)
        
    # while not found and have elements in array 
    # (if left and right pointers cross, then exhausted elements to search)
    while left <= right:

        # compare value to element in middle of array
        if value == input_array[middle]:
            return middle
            
        # if greater, search second half of array by updating the lower = middle + 1 (since not equal to middle)
        elif value > input_array[middle]:
            left = middle + 1
        
        # else search first half by updating right to = middle - 1 (since not equal to middle)
        else:
            right = middle - 1
        
        middle = (right - left)//2 + left
    
    return -1



#%% Test ####

if __name__ == '__main__':
    test_list = [1,3,9,11,15,19,29]
    test_val1 = 25
    test_val2 = 15
    print (binary_search(test_list, test_val1))
    print (binary_search(test_list, test_val2))