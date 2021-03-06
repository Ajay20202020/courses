"""The LinkedList code from before is provided below.
Add three functions to the LinkedList.
"get_position" returns the element at a certain position.
The "insert" function will add an element to a particular
spot in the list.
"delete" will delete the first element with that
particular value.
Then, use "Test Run" and "Submit" to run the test cases
at the bottom."""

class Element(object):
    def __init__(self, value):
        self.value = value
        self.next = None
        
class LinkedList(object):
    def __init__(self, head=None):
        self.head = head
        
    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
        else:
            self.head = new_element
            
    def get_position(self, position):
        """Get an element from a particular position.
        Assume the first position is "1".
        Return "None" if position is not in the list."""
        
        current = self.head
        
        i = 1
        
        while i < position:
            
            if current.next:
                current = current.next
                i+=1
            else:
                return None
        
        return current
    
    def insert(self, new_element, position):
        """Insert a new node at the given position.
        Assume the first position is "1".
        Inserting at position 3 means between
        the 2nd and 3rd elements."""
        
        if position == 1:   
            self.head, new_element.next = new_element, self.head
        else:
            element_before = self.get_position(position - 1)
        
            #element_after = element_before.next # if none then new element next = None
            #element_before.next = new_element
        
            # if there is an element in the position we want to insert, 
            # then set next reference of new element to the current elmenet
            #new_element.next = element_after 
            
            element_before.next, new_element.next = new_element, element_before.next

    
    def delete(self, value):
        """Delete the first node with a given value."""
        current = self.head
        
        match_found = False
        
        if current.value == value:
            match_found = True
            self.head = self.head.next
        
        while not match_found and current.next:
            if current.next.value == value:
                match_found = True
                current.next = current.next.next
            
            current = current.next
    
#%% Test code ####
    
if __name__ == '__main__':

    # Test cases
    # Set up some Elements
    e1 = Element(1)
    e2 = Element(2)
    e3 = Element(3)
    e4 = Element(4)
    
    # Start setting up a LinkedList
    ll = LinkedList(e1)
    ll.append(e2)
    ll.append(e3)
    
    # Test get_position
    # Should print 3
    print (ll.head.next.next.value)
    # Should also print 3
    print (ll.get_position(3).value)
    
    # Test insert
    ll.insert(e4,3)
    # Should print 4 now
    print (ll.get_position(3).value)
    
    # Test delete
    ll.delete(1)
    # Should print 2 now
    print (ll.get_position(1).value)
    # Should print 4 now
    print (ll.get_position(2).value)
    # Should print 3 now
    print (ll.get_position(3).value)