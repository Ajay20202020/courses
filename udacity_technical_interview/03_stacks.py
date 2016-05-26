"""Add a couple methods to our LinkedList class,
and use that to implement a Stack.
You have 4 functions below to fill in:
insert_first, delete_first, push, and pop.
Think about this while you're implementing:
why is it easier to add an "insert_first"
function than just use "append"?"""

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

    def insert_first(self, new_element):
        # set next reference of new element to current head, then change 
        # the current head to the new element
        self.head, new_element.next  = new_element, self.head

    def delete_first(self):
        
        deleted_element = self.head
        
        # set the head to the current next reference of the head if head is not none (no elements in list)
        if self.head:
            self.head = self.head.next
     
        return deleted_element

class Stack(object):
    def __init__(self,top=None):
        self.ll = LinkedList(top)

    def push(self, new_element):
        self.ll.insert_first(new_element)

    def pop(self):
        return self.ll.delete_first()

#%% Test #### 
if __name__ == '__main__':
    # Test cases
    # Set up some Elements
    e1 = Element(1)
    e2 = Element(2)
    e3 = Element(3)
    e4 = Element(4)
    
    # Start setting up a Stack
    stack = Stack(e1)
    
    # Test stack functionality
    stack.push(e2)
    stack.push(e3)
    print (stack.pop().value) #3
    print (stack.pop().value) #2
    print (stack.pop().value) #1
    print (stack.pop()) # None
    stack.push(e4)
    print (stack.pop().value) #4