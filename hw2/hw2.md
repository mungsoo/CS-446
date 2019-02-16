# CS-446

Machine Learning
#### 1 Singular Value Decomposition and norms
(a) Thin SVD of A, $A = USV^T$. Then rewrite  x in terms of V, $x = Vy$. 
Note that if $||x||_2 = x^Tx = y^TV^Ty = ||y||_2 \leqq 1$.
So we have 
$$||Ax||_2 =||USV^TVy||_2 = ||USy||_2 = \sqrt{y^TS^TSy}$$
Assume the singular values of A is $\lambda_i, 0 < i \leqq r$, then we have
$$\sqrt{y^TS^TSy} = \sqrt{\sum_i=1^r \lambda_i^2y_i^2} s.t. \sum_i=1^ry_i^2\leqq 1$$
So $\max_{||x||_2\leqq1}||Ax||_2 = \max \lambda_i$.
