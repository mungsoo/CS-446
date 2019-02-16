# CS-446

Machine Learning
#### 1 Singular Value Decomposition and norms
(a) Thin SVD of A, $A = USV^T$. Then rewrite  x in terms of V, $x = Vy$. 
Note that $||x||_2 = x^Tx = y^TV^TVy = ||y||_2 \leq 1$.
So we have 

$$ ||Ax||_2 =||USV^TVy||_2 = ||USy||_2 = \sqrt{y^TS^TSy}$$

Assume the singular values of A are $\lambda_i, 0 < i \leq r$, then we have 

$$\sqrt{y^TS^TSy} = \sqrt{\sum_{i=1}^r \lambda_i^2y_i^2} $$

$$s.t. \sum_{i=1}^ry_i^2\leq 1$$

So $\max_{||x||_2\leq1}||Ax||_2 = \max \lambda_i$.


(b)

$$ ||A||_2 = max ||Ax||_2 where ||x||_2=1$$

$$ \geq ||A \frac{x}{||x||_2}||_2 $$ 

$$= \frac{||Ax||_2}{||x||_2} $$

Hence, we have $||A||_2||x||_2 \geq ||Ax||_2$

(c)
Assume A = $\left[begin{array}{cccc} 
a_1&a_2 &\cdots &a_d 
\end{array}\right]$

