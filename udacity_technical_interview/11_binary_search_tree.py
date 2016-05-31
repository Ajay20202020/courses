class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST(object):
    def __init__(self, root):
        self.root = Node(root)

    def insert(self, new_val):
        
        current = self.root
        match = False
        
        # iterate until either there is a match or hit a leaf
        while not match and current:
            
            if new_val == current.value:
                match = True
                print(str(new_val) + "already in tree. Not inserted")
            
            elif new_val > current.value:
                current = current.right
                
            else:
                current = current.left
        
        # if hit a leaf and value is not in tree, add node 
        if not match:
            current = Node(new_val)
        

    def search(self, find_val):
        
        current = self.root
        
        while current:
            
            if find_val == current.value:
                return True
            
            elif find_val > current.value:
                current = current.right
                
            else:
                current = current.left
    
        return False
    
#%% Test
    
if __name__ == '__main__':
        
    # Set up tree
    tree = BST(4)
    
    # Insert elements
    tree.insert(2)
    tree.insert(1)
    tree.insert(3)
    tree.insert(5)
    
    # Check search
    # Should be True
    print (tree.search(4))
    # Should be False
    print (tree.search(6))