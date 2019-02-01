# CS-446

Machine Learning
#### 1 Decision Trees
(a) Assume the horizontal axis is x, vertical axis is y. Then if x > 2, label = blue. If x <= 2, label = red.

(b) If x <= 2, label = red. If X > 2 and y < 2, label = red. If x > 2 and y >= 2, label = blue.

(c) Same as (b). If x <= 2, label = red. If X > 2 and y < 2, label = red. If x > 2 and y >= 2, label = blue.

#### 2 Linear Regression

![](https://github.com/mungsoo/CS-446/blob/master/image/Q1.3.jpg?raw=true)

#### 3 Singular Value Decomposition
(a)If $s_i \neq 0$  for all $i \in \\{1,...,n\\}$, then we have $A^{+} = VS^{+}U^{\top}$, hence
$$AA^{+} = UU^{\top}$$
where U $\in R^{n \times n}$. Because $U$ is orthonormal matrix, then we have $$U^{\top}U = UU^{\top} = I$$
Hence, $AA^{+}=I$. So $A$ is invertable.
If $A$ is invertable, then pseudo-inverse $A^+ = A^{-1}$. So $$AA^+ = I$$
$$UU^{\top} = I$$
So we get $U$ is orthonormal square matrix, whose rank is n. So for all $i \in \\{1,...,n\\}$,  $s_i \neq 0$

(b)First, compute full SVD of $A$, we get $A = USV^\top$, then we have
$$A^{\top} A = VS^{\top} SV^{\top}$$
where $V \in R^{d \times d}$ and $V$ is orthonormal, $S^\top S \in R^{d \times d}$. Hence, $$(A^\top A) + \lambda I = V (S ^{\top} S + \lambda I)
V^{\top}$$
Obviously, $(S ^{\top} S + \lambda I)$ is a $d \times d$ diagonal matrix whose diagonal elements all greater than 0. According to (a), we can conclude that 
$V (S ^{\top} S + \lambda I)
V^{\top}$ is invertable. So the matrix $(A^\top A) + \lambda I$ is invertable.

(c)$$(A^\top A + \lambda I)^{-1} A^\top = V (S ^\top S + \lambda I)^+ S^\top U^\top$$
as $\lambda$ approach 0, $(S ^\top S + \lambda I)^+ S^\top$ converge to $S^+ (S^\top)^+ S^\top = S^+$.
So $$lim_{\lambda \rightarrow 0} (A^\top A + \lambda I)^{-1} A^\top = VS^+ U^\top = A^+$$

#### 4 Polynomial Regression
(a)
$$\phi (x) =
\left[
\begin{array}
{cccccccccc}
1&x_1&x_2&x_3&x_1x_1&x_1x_2&x_1x_3&x_2x_2&x_2x_3&x_3x_3
\end{array}
\right]^\top
$$

(d)

![](https://github.com/mungsoo/CS-446/blob/master/image/Q4.4.jpg?raw=true)

(e)
The output of poly_xor() is

![](https://github.com/mungsoo/CS-446/blob/master/image/Q4.5-result.jpg?raw=true)

So the Linear Regression doesn't give the correct solution. The polynomial Regression correctly classify all points.

I try to plot the contour of each method. However, it showed "No contour found in given range" when I feed contour_plot() with Linear Regression function. So I just plot the contour of Polynomial Regression.

![](https://github.com/mungsoo/CS-446/blob/master/image/Q4.5.jpg?raw=true)

#### 5 Nearest Neighbour
(b)
Original data

![](https://github.com/mungsoo/CS-446/blob/master/image/1nn_data.jpg?raw=true)

Voronoi diagram

![](https://github.com/mungsoo/CS-446/blob/master/image/1nn_voronoi.jpg?raw=true)

(c)
The accuracy is 0.97777

![](https://github.com/mungsoo/CS-446/blob/master/image/nn_acc.jpg?raw=true)

#### 6 Logistic Regression
(a)


$$\frac{\partial R}{\partial w} =- \frac{1}{n} \sum_{i=1}^{n}\frac{exp(-y_iw^\top x_i)x_i}{1+exp(-y_iw^\top x_i)}=-\frac{1}{n}X^{'\top} M$$
where $M \in R^{n\times 1}$ with i th elements $\frac{exp(-y_iw^\top x_i)}{1+exp(-y_iw^\top x_i)} $. $X^{'} \in R^{n\times d}$ is 
$$\left[
\begin{array}
{cccc}
y_1x_1& y_2x_2 & \cdots &y_nx_n
\end{array}
\right]^\top
$$ 


So the gradient update rule for empirical risk is:

For $k = 0, 1, ...,$:

$w^{'} = w + \eta \frac{1}{n} \sum_{i=1}^{n}\frac{exp(-y_iw^\top x_i)x_i}{1+exp(-y_iw^\top x_i)}$

$w = w^{'}$


(c)

![](https://github.com/mungsoo/CS-446/blob/master/image/log_vs_ols.jpeg?raw=true)

The logistic regression is better. First, the hyperplane of logistic regression seems to approximately maximize the margin. To maximize the margin is the one of the core ideas of SVM and it works well in many problems. Second, this is a classification task. Although the linear regression seperates the data set correctly, it just tries to fit the label of each point, which is either 1 or -1.The hyperplane of is the projection of $w_1x_1+w_2x_2+b=0$ on x plane. Although it is somewhat reasonable, it is not as good as logistic regression.