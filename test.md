# CS231n - Lecture 2

## Concept
- find <img src="/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/> which has parameter(or weight) <img src="/tex/e0d75638341aaa771a47999137d21473.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/>

<p align="center"><img src="/tex/40852839dbbdb0390f3fb896634b494f.svg?invert_in_darkmode&sanitize=true" align=middle width=286.73515695pt height=16.438356pt/></p>
- Example :

<p align="center"><img src="/tex/8d2381a80ac4dfc31b5d5afbc0ff0a16.svg?invert_in_darkmode&sanitize=true" align=middle width=764.37682365pt height=78.9048876pt/></p>

- (생각할 것) 보통 확률로 Classification을 수행하는데, score를 이용함으로서 classification 한다. (?)
- (생각할 것) bias라 함은 error와 다른 개념? 일반적인 선형회귀에서 <img src="/tex/3bde0199092dbb636a2853735fb72a69.svg?invert_in_darkmode&sanitize=true" align=middle width=15.85051049999999pt height=22.831056599999986pt/> 와 같은 역할을 하는 것인가?

- Linear Classification은 기본적으로 Decision Boundary를 만들어서 Classification 하는 문제와 같음. 각각 하나의 linear boundary를 가진다고 보면 됨. 위의 예시에서는 세개의 boundary가 생김
- 이러한 Linear Classification 을 쌓아올리면 Neural Network 를 생성할 수 있음.
- e.g. 이미지와 그 설명이 함께 제공되는 데이터라면 이미지는 CNN으로, 설명은 RNN으로 처리하여 Neural Network를 만들수도 있음.
- 이러한 Linear Model 의 문제는 역시 선형관계만 발견이 가능하다는 점으로, 비선형 관계에 대한 탐색은 불가능함.
- 그렇다면 <img src="/tex/0b367a14bbbb72a86273badd4408e55b.svg?invert_in_darkmode&sanitize=true" align=middle width=29.429203649999987pt height=22.831056599999986pt/>를 찾는 방식? with Loss Function, Optimization, Stochastic Gradient Descent
