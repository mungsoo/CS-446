# CS-446

Machine Learning
#### 1 Singular Value Decomposition and norms
(a) Thin SVD of A, $A = USV^T$. Then rewrite  x in terms of V, $x = Vy$. 
Note that $||x||_2 = x^Tx = y^TV^TVy = ||y||_2 \leq 1$.
So we have 

$$ ||Ax||_2 =||USV^TVy||_2 = ||USy||_2 = \sqrt{y^TS^TSy}$$

Assume the singular values of A are $s_i, 0 < i \leq r$, then we have 

$$\sqrt{y^TS^TSy} = \sqrt{\sum_{i=1}^r s_i^2y_i^2} $$

$$s.t. \sum_{i=1}^ry_i^2\leq 1$$

So $\max_{||x||_2\leq1}||Ax||_2 = \max s_i$.\\


(b)
$$||A||_2 = \max_{||x||_2\leq1} ||Ax||_2 \geq ||A \frac{x}{||x||_2}||_2 = \frac{||Ax||_2}{||x||_2} $$

Hence, we have $||A||_2||x||_2 \geq ||Ax||_2$\\

(c)
Assume A = $\left[\begin{array}{cccc} 
a_1&a_2 &\cdots &a_d
\end{array}\right]$


Then $$A^TA = \left[\begin{array}{cccc} 
a_1^Ta_1&a_1^Ta_2 &\cdots &a_1^Ta_d \\\\
a_2^Ta_1&a_2^Ta_2 &\cdots &a_2^Ta_d \\\\
\vdots &\vdots & \ddots & \vdots\\\\
a_d^Ta_1 &a_d^Ta_2 & \cdots &a_d^Ta_d
\end{array}\right]
$$


$$tr(A^TA) = \sum_{i=0}^{d} a_i^Ta_i= \sum_{i=1}^{d}\sum_{j=1}^{d}A_{ij}^2$$

$Q.E.D$\\

(d)Write $A$ as $A = USV^T$, then we have
$$ tr(A^TA) = tr(VS^TU^TUSV^T) = tr(VS^TSV^T) = \sum_{i=1}^n\sum_{j=1}^rs_j^2 v_{ij}^2 = \sum_{j=1}^r s_j^2 \sum_{i=1}^n v_{ij}^2 = \sum_{j=1}^r s_j^2$$

Hence, $$||A||_F^2 = \sum_{j=1}^r s_j^2$$

(e)
First, we proof that
 $$\max_{||x||_2\leq1} ||ABx||_2 \leq ||A||_2||Bx||_2\leq ||A||_2 ||B||_2 ||x||_2 = ||A||_2 ||B||_2$$

Then we have

$$||A||_2^2 ||B||_F^2 = 
\sum_{i=1}^m||A||_2^2||B:,i||_2^2 \geq   \sum_{i=1}^m ||AB;,i||_2^2 =||AB||_F^2$$\\

(f)
Thin SVD:
$$A  = USV^T$$ 
Pseudo-inverse:
$$ A^+ = VS^+U^T$$
Hence, we have $$A^+A = VS^+U^TUSV^T = VV^T= VI_rV^T$$
$$ AA^+ = USV^TVS^+U^T=UU^T = UI_rU^T$$
Hence, $$||AA^+||_2 = ||UU^T||_2 = \max s_i = 1$$
$$A^+A||_2 = ||VV^T||_2 = \max s_i = 1 = ||AA^+||$$

We also have
$$ ||A^+A||_F = ||VV^T||_F = \sqrt{\sum_{i=1}^r s_i} = \sqrt{r} = ||AA^+||_F$$


(g)
Assume $n\times r$ orthogonal matrix $V$ and $U$, and the vector $v$ satisfies $V^TX =0$ and $U^Tx \neq 0$, then construct $A = UI_rV^T$.\\
