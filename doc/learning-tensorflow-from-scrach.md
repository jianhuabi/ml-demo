# Learning Tensorflow by building it from Scratch

I've been straggling to learn Tensorflow with zero prior knowledge which is a painful experience. I was a programmer from OOP land and Tensorflow programming methodology is alien to me. So I started to search for blogs and youtube videos, also technical books to get me comfortable with Tensorflow but they either suggest to learn Tensorflow higher level API like Keras or jump right into Tensorflow core API yet focus on ML model explanation. I mean they are great but I believe knowing just Tensorflow high-level API isn't enough for you to be comfortable with Tensorflow, my experience was that sooner or later you definitely need to rewind back to basics. Perhaps I could build a tiny Tensorflow library from scratch to simulate real Tensorflow internal structures in a nutshell. The source code I made was inspired by some tech blog, huge thanks to the author.

## Build Tensorflow from Scratch

Let's see a simple Tensorflow program which calculates a linear function.

**y = m*x + b**

```python=
import tensorflow as tf
import numpy as np
import google.datalab.ml as ml
import matplotlib.pyplot as plt

x_data = np.linspace(0.0, 10.0, 100, dtype=np.float32)

xph = tf.placeholder(dtype=tf.float32, shape=[len(x_data)], name='x')
cons_m = tf.constant(dtype=tf.float32, value=0.35, name='m')
cons_b = tf.constant(dtype=tf.float32, value=2.0, name='b')

model = tf.add(tf.multiply(xph, cons_m, name='mul'), cons_b, name='add')

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  
  y_data = sess.run(model, feed_dict={xph:x_data})
  
plt.plot(x_data,y_data,'r')
```


![](https://i.imgur.com/JxhT4bJ.png)

The code has nothing to do with ML yet the coding style is essential for any useful Tensorflow program. Let's build our own tiny Tensorflow that implement the same function as googles.
Machine Learning is all about mathematical calculation. Tensorflow models mathematical calculus as a Graph. For example, linear calculus **y=m*x + b** can be model as a graph in below.

![](https://i.imgur.com/d6gp2Xh.png)

We defined a Python class to model a graph.
The graph maintains lists of **operations, placeholders, and variables**. 

Variable in Tensorflow represents training model parameters yet to simplify our codes, I made Variable as well to represent Tensorflow constant.

Next, let's build the Graph to model the linear calculus.

Operation class models calculus operation in our training model, like add, minus, multiply, matrix dot product etc, etc.

We defined a Python class to model a graph.


```python=

class Graph():
    
    def __init__(self):
        
        self.operations = []
        self.placeholders = []
        self.variables = []
        
    def set_as_default(self):
        """
        Sets this Graph instance as the Global Default Graph
        """
        global _default_graph
        _default_graph = self
```

The graph maintains lists of operations, placeholders, and variables. ***Variable*** in Tensorflow represents training model parameters yet to simplify our codes, I made Variable as well to represent Tensorflow ***constant***.

Next, let’s build the Graph to model the linear calculus.


```python=
class Operation():
    """
    An Operation is a node in a "Graph". TensorFlow will also use this concept of a Graph.
    
    This Operation class will be inherited by other classes that actually compute the specific
    operation, such as adding or matrix multiplication.
    """
    
    def __init__(self, input_nodes = []):
        """
        Intialize an Operation
        """
        self.input_nodes = input_nodes # The list of input nodes
        self.output_nodes = [] # List of nodes consuming this node's output
        
        # For every node in the input, we append this operation (self) to the list of
        # the consumers of the input nodes
        for node in input_nodes:
            node.output_nodes.append(self)
        
        # There will be a global default graph (TensorFlow works this way)
        # We will then append this particular operation
        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)
  
    def compute(self):
        """ 
        This is a placeholder function. It will be overwritten by the actual specific operation
        that inherits from this class.
        
        """
        
        pass
```

Operation class models calculus operation in our training model, like add, minus, multiply, matrix dot product etc, etc.

It memorizes its input_nodes (line: 13). So when Graph’s running engine performs ***Compute*** operation, the engine will inject computing inputs from these input nodes to an Operation Object which then computes outputs based on these values. We will revisit this operation when we build the compute engine.

It then adds itself as an output_nodes in each input_nodes. ***That is the beauty of how calculus is modeled as a linked Operations by python code.***

The following listings implement several common operations like add, multiply and matrix dot operation.

```python=

class add(Operation):
    
    def __init__(self, x, y):
         
        super().__init__([x, y])

    def compute(self, x_var, y_var):
         
        self.inputs = [x_var, y_var]
        return x_var + y_var
```

Let’s say we call ***add(m, b), the (m,b)*** are **input_nodes** in our case so this call will create a graph like below.

![](https://i.imgur.com/t6ZE8SA.png)

And ***when graph engine performs add.compute, the engine will inject input values [x_var, y_var] from input_nodes which is [m, b].***

Ok, so far so good, Let’s implement our other Operations.

```python=
class multiply(Operation):
     
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a_var, b_var):
         
        self.inputs = [a_var, b_var]
        return a_var * b_var
      
class matmul(Operation):
     
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a_mat, b_mat):
         
        self.inputs = [a_mat, b_mat]
        return a_mat.dot(b_mat)
```

With those Operations, we basically can implement a compute graph as we showed here.

**y = m*x + b**

![](https://i.imgur.com/CfZfdZw.png)

Some may ask, yes, we modeled calculus operations but ***what about (x, m, b),*** how can we model them in python?

The ***x*** is the input of our model. We don’t know its value yet until we run the calculation. So in Tensorflow, the class Placeholder is used to model them.

```python=
class Placeholder():
    """
    A placeholder is a node that needs to be provided a value for computing the output in the Graph.
    """
    
    def __init__(self):
        
        self.output_nodes = []
        
        _default_graph.placeholders.append(self)
```

As you can see, in our implementation, a placeholder only maintains ***outputnodes*** list and register itself to ***default graph***.

Similar to placeholder, we implement Variable class in python.

```python=
class Variable():
    """
    This variable is a changeable parameter of the Graph.
    """
    
    def __init__(self, initial_value = None):
        
        self.value = initial_value
        self.output_nodes = []
        
         
        _default_graph.variables.append(self)
```

A Variable in our implementation memorizes its input value to ***self.value (line 8).*** It as well maintains a list of output_nodes and registers itself to default graph as Placeholder does.

At this point, we have implemented all graph building primitives but without a graph engine to drive the calculation.

OK, Let’s build our own compute engine!

```python=
class Session:
    
    def run(self, operation, feed_dict = {}):
        """ 
          operation: The operation to compute
          feed_dict: Dictionary mapping placeholders to input values (the data)  
        """
        
        # Puts nodes in correct order
        nodes_postorder = traverse_postorder(operation)
        
        for node in nodes_postorder:

            if type(node) == Placeholder:
                
                node.output = feed_dict[node]
                
            elif type(node) == Variable:
                
                node.output = node.value
                
            else: # Operation
                
                node.inputs = [input_node.output for input_node in node.input_nodes]

                 
                node.output = node.compute(*node.inputs)
                
            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)
        
        # Return the requested node value
        return operation.output
```

Take ***y = mx + b*** as an example, the y will be the input to Session.run function. This function will drive calculus according to the graph we build so far.

The acute reader might notice this alien function (traverse_postorder) at line 10. The implementation is listed below and it is used to output a correct compute order for example multiply should be computed first and then add operation. I don’t pretend I fully understand the implementation and it doesn’t bother us at all to understand Tensorflow modeling.

![](https://i.imgur.com/dwwC2Tl.png)

So, the run function basically goes over compute graph’s node one by one and compute or prepare its output values based on if the node is a Placeholder or an Operation.

If the current node is a Placeholder (line 14), it simply memorizes the inputs values as its output from a python dict structure. Similar to Placeholder, if a node is a Variable like [line 18], it as well memorize the input's value as its output.

Now, the real calculation happens if a node is a type of Operation [line 24 to 27]. It first retrieves computing values from the operation node’s input_nodes. If you rememberers, the operation node’s input_nodes are either Placeholder/Variable which output is already prepared or another operation whose output is calculated before this node. Think second time if it confuses you.

Then, the compute engine injects all inputs to the node’s compute function to calculate the node’s output.

The engine loop through every node in the default graph and returns the final output from the last node in the graph.

Let’s see how we use all these primitives to run a simple linear algebra.

```python=
g = Graph()
g.set_as_default()
A = Variable([[10,20],[30,40]])
b = Variable([1,1])
x = Placeholder()
y = matmul(A,x)
z = add(y,b)
sess = Session()
result = sess.run(operation=z,feed_dict={x:10})
final result is: 
array([[101, 201],        
[301, 401]])
```
Isn’t that amazing and how similar it dose as Tensorflow !!!!
