# Miscellaneous

## Proof probability and score

We are going to demonstrate the link between optimizing the logloss for a risk score model without ordering constraint, and calibration.

Suppose that there are $n$ samples, with their associated binary target $y_i \in \{0,1\}, i\in [1,...,n]$ and scores $s_i i\in [1,...,n]$. The scores $s_i$ are a sum of small relative integers, therefore we can enumerate all possible score values as $\{s_1,...,s_k \}$.

Let $p_{s_j} \in (0,1), j \in [1,k]$ be the probability associated to score $s_j$.

The log loss associated to the $n$ samples with probability $p_j, j \in [1,k]$ is equal to :

$L = \sum_{i=1}^n -y_{i}log(p_{s_i}) - (1-y_i)log(1-p_{s_i})$ 

We denote respectively by $n_j^+ = \sum_{i=1}^n \mathbb{1}_{s_i=s_j}y_i$, and $n_j^- = \sum_{i=1}^n \mathbb{1}_{s_i=s_j}(1-y_i)$ the number of positive and negative samples at each possible score value $s_j, j \in [1,...,k]$.

We can rewrite $L$ with $n_j^+$ and $n_j^-$ 

$L(p_{s_1},...,p_{s_k})=\sum_{j=1}^k-n_j^+log(p_{s_j})- n_j^-log(1-p_{s_j}) = \sum_{j=1}^kL_j$

Each $L_j$ is a convex function on $p_{s_j} \in (0,1)$, as it's a nonnegative weighted sum of two convex functions. Thus, $L$ is also convex as a nonnegative weigthed sum of convex functions.

Since $L$ is convex, there exists only one minimum, where the gradient of $L$, denoted by \nabla L(p_{s_1},...,p_{s_k}), is equal to 0.

$$\nabla L(p_{s_1},...,p_{s_k}) = \left[\begin{array}{c} 
\dfrac{\partial L}{\partial p_{s_1}}(\left.p_{s_1},...,p_{s_k}\right)\\
\vdots \\
\dfrac{\partial L}{\partial p_{s_k}}(\left.p_{s_1},...,p_{s_k}\right)\\
\end{array}\right] = \left[\begin{array}{c} 
\dfrac{\partial L_1}{\partial p_{s_1}}(\left.p_{s_1}\right)\\
\vdots \\
\dfrac{\partial L_k}{\partial p_{s_k}}(\left.p_{s_k}\right)\\
\end{array}\right]
$$
$$\nabla L(p_{s_1},...,p_{s_k}) = \left[\begin{array}{c} 
-\frac{n_1^+}{p_{s_1}}+ \frac{n_1^-}{1-p_{s_1}}\\
\vdots \\
-\frac{n_k^+}{p_{s_k}}+ \frac{n_k^-}{1-p_{s_k}}\\
\end{array}\right] 
$$

The probabilities $p_{s_j}, j \in [1,k]$ minimizing the logloss satisfies the folowing system:
$$
\left\{\begin{array}{rcl} 
-\frac{n_1^+}{p_{s_1}}+ \frac{n_1^-}{1-p_{s_1}} & = & 0\\
\vdots & = & \vdots\\
-\frac{n_k^+}{p_{s_k}}+ \frac{n_k^-}{1-p_{s_k}}& = & 0\\
\end{array}\right.
$$

which leads to

$$
\left\{\begin{array}{rcl} 
p_{s_1} & = & \frac{n_1^+}{n_1^-+n_1^+}\\
\vdots & = & \vdots\\
p_{s_k}& = & \frac{n_k^+}{n_k^-+n_k^+}\\
\end{array}\right.
$$

We see that optimizing the logistic loss without the ordering constraint leads to probabilites perfectly calibrated on the given samples, as it is exactly the proportion of positive samples for each score on the total of total samples for each score.



