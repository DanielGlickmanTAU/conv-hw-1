r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer: Increasing the value of k will not necessarily increase the accuracy. 
Let's look at the output graph above. The best accuracy is achieved at k=15. But when k=50
the accuracy decreases. 

As it can be seen in the graph, we would like k to be somewhere in the middle for possibly
the following reasons:
- We would not want k to be "too small" since we might have some examples whose closest
neighbours are the wrong classes, so increasing a bit k might generalize better

- On the other hand, increasing k too much might be bad for generalization for example if we chose
k to be larger than the number of examples of a specific category, we might find other categories 
to be closer, and thus mislabel such unseen examples.
**
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
** The delta forces a distances of the samples from the hyperplane W creates. 
The same hyperplane will be reached, but scaled by a constant scalar, which will scale the weights by the same constant**
"""

part3_q2 = r"""
**We can think of the weights for each class(columns) as a vector representing the "average" 
x example for that class. the class of x is scored by a dot product with x, so the weight will
try to embody a close representation,in the vector space, to x's of that class.

The distance analogy is similar with knn but different in that:
1) different distance metric is used(dot product vs euclidean)**
2) knn is optimized/punished for distance with the top k examples while the linear classifier measures
the average distance from all the dataset
"""

part3_q3 = r"""
**
 1) When the learning rate is too high I expect the training loss to not converge at all,
 or be very noisy, i..e jump up and down. Clearly this is not the case as the training loss decreases with time
 When the learning rate is too low, the loss will decrease but at a lower rate, and the final loss would not be 
 as high as possible within the same number of epochs.
 
 It is possible that slightly higher learning rates would lead to lower overall loss but it seems
 like overall the learning rate is good as the loss decreases nicely and approches zero.
 
 2) The model is slightly overfitted. We can see that by the fact that the train accuracy is higher
 then the validation's. That fact remains stable from about epoch 10 
**
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
** Ideally we would like the residiual plot to be centered around y=0 line. If it's exactly on the y=0
line then that means we have estimated precisely for that specific data point. Clearly, to have data points on this
line is unlikely, but the ideal situation would be for the data points to be as close as possible to y=0 line.
Furthermore, we would like the dotted lines to be closer to zero. In the plot we can see that our model, with the
chosen parameters generalized well.

"""

part4_q2 = r"""
**Your answer:**
1. The use of np.logspace might be to do the grid search on the regularization paramater on a wider range.



2. The number of times the model was fitted is:

        |degree_range| x |lambda_range| x k 

    Where k is the k-fold parameter.  The reason for that is we try each possible
    combination of degree and lambda, and since we train each model k times we get
    the expression above.
"""

# ==============
