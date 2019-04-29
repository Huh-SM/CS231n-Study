# CS231n - Lecture 2

## Concept
- find $f(x)$ which has parameter(or weight) $ w $

$$\text{Input} \rightarrow f(x, W) = Wx \rightarrow  \text{output score}$$
- Example :

$$
\begin{bmatrix}
    0.2 & -0.5 & 0.1 & 2.0 \\
    1.5 & 1.3 & 2.1 & 0.0 \\
    0.0 & 0.25 & 0.2 & -0.3 
\end{bmatrix}\begin{bmatrix}
    56 \\
    231 \\
    24 \\
    2
\end{bmatrix} + \begin{bmatrix}
    1.1 \\
    3.2 \\
    -1.2 \\
\end{bmatrix} = \begin{bmatrix}
    -96.8 & \text{(Cat Score)} \\
    437.9 & \text{(Dog Score)} \\
    61.95 & \text{(Ship Score)} \\
\end{bmatrix}  \\
\\

\\
\\
W(weight)X(input) + b(bias) = Score

$$

- (생각할 것) 보통 확률로 Classification을 수행하는데, score를 이용함으로서 classification 한다. (?)
- (생각할 것) bias라 함은 error와 다른 개념? 일반적인 선형회귀에서 $\beta_0$ 와 같은 역할을 하는 것인가?

- Linear Classification은 기본적으로 Decision Boundary를 만들어서 Classification 하는 문제와 같음. 각각 하나의 linear boundary를 가진다고 보면 됨. 위의 예시에서는 세개의 boundary가 생김
- 이러한 Linear Classification 을 쌓아올리면 Neural Network 를 생성할 수 있음.
- e.g. 이미지와 그 설명이 함께 제공되는 데이터라면 이미지는 CNN으로, 설명은 RNN으로 처리하여 Neural Network를 만들수도 있음.
- 이러한 Linear Model 의 문제는 역시 선형관계만 발견이 가능하다는 점으로, 비선형 관계에 대한 탐색은 불가능함.
- 그렇다면 $ W,b $를 찾는 방식? with Loss Function, Optimization, Stochastic Gradient Descent
