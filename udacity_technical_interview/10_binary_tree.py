
class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree(object):
    def __init__(self, root):
        self.root = Node(root)

    def search(self, find_val):
        """Return True if the value
        is in the tree, return
        False otherwise."""
        return self.preorder_search(self.root, find_val)

    def print_tree2(self):
        """Print out all tree nodes
        as they are visited in
        a pre-order traversal."""
        return self.preorder_print2(self.root).strip("-")

    def preorder_search(self, start, find_val):
        """Helper method - use this to create a 
        recursive search solution."""
        
        # base case = node is None (reach a leaf)
        if not start:
            return False
        
        # base case = value of node equals value searching
        if start.value == find_val:
            return True
            
        # recursive on start= children 
        return self.preorder_search(start.left,find_val) or self.preorder_search(start.right, find_val)

    def preorder_print2(self, start):
        """Helper method - use this to create a 
        recursive print solution."""
        
        # base case = node is None
        if not start:
            return ""
            
        # recursive on start= children 
        traversal = str(start.value) + "-" + str(self.preorder_print2(start.left)) + str(self.preorder_print2(start.right) )      
            
        return traversal

#%% Test Main

# 

# Set up tree
tree = BinaryTree(1)
tree.root.left = Node(2)
tree.root.right = Node(3)
tree.root.left.left = Node(4)
tree.root.left.right = Node(5)

# Test search
# Should be True
print (tree.search(4))
# Should be False
print (tree.search(6))

# Test print_tree
# Should be 1-2-4-5-3
print (tree.print_tree2())