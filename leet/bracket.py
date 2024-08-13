#bracket.py

def bracket(inputs):
    stack = Stack()
    
    for idx in range(len(inputs)):
        input = inputs[idx]
        if input in ['{', '(']:
            stack.push(input) 
        else:
            pop = stack.pop()
            if pop == '{':
                if input != '}':
                    return False                    
            if pop == '(':
                if input != ')':
                    return False
    if len(stack.inputs) > 0:
        return False
    return True
        


class Stack:
    inputs = []
    def push(self, input):
        self.inputs.append(input)
    
    def pop(self):
        pop = self.inputs[len(self.inputs)-1]
        self.inputs = self.inputs[:-1]
        return pop 
    
res  = bracket(['{','(','{','}',')'])
print(res)