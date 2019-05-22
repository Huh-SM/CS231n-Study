# Lecture 3
___
### Lecture 2 에 이어서.
- Linear Classification을 통해 Score function을 생성함.
- score를 계산하는 fucntion의 parameter $w$(weight)를 어떻게 찾을까? by Loss function, how? optimization!

___
## Loss function
- 경우에 따라 Cost fucntion으로 불리기도 함.
- loss(손실)을 최소화하면 되는거 아니냐의 idea
- 회귀에서도 error의 추정량의 제곱을 최소화 시키는 방법을 쓰듯..
- Loss over datasets 
$$ L = \frac{1}{N}\sum_i L_i(f(x_i, W), y_i) $$

___
### Multiclass Support Vector Machine loss (Multiclass SVM loss)
- 이 손실함수의 idea는 어느 정도 허용수준($\Delta$)이상으로 정답 클래스의 score를 더 크게 하는 방향으로 설계한다는 것.(good방향)
	- $\Delta$ 는 hyper-parameter로 정해줘야하는 부분이다.
- formula
$$ L_i = \sum_{j\neq y_i}\max(0, s_j-s_{y_i} + \Delta)$$
- 만약 계산된 3개의 사진에서 고양이, 개, 개구리인지 판단하는 스코어가 각각 $S_1 = [3.2, 5.1, -1.7]$ $S_2 = [1.3, 4.9, 2.0]$ $S_3 = [2.2, 2.5, -3.1]$ $\Delta = 1$ 이라고 하면, 
 $ L_1 = max(0, 5.1-3.2+1) + max(0, -1.7-3.2+1) = 2.9$ 
 $L_2 = max(0, 1.3-4.9+1) + max(0, 2.0 - 4.9 + 1) = 0 $ 
 $L_3 = max(0, 2.2+3.1+1) + max(0, 2.5+3.1+1) = 12.9$  
 $ L = (2.9 + 0 + 12.9)/3 = 5.27 $ 
 
- Multi-SVM loss에 대해서 생각해야할 것이 몇가지 있는데, formula에 근거해서 생각하면 됨.
- 만약 정답 카테고리의 score가 기존의 스코어의 순서를 바꾸지 않는 한에서 변하면 loss의 변화는? $\rightarrow$ loss는 변하지 않는다.
- Multi SVM loss의 가능한 값의 영역은? $\rightarrow$ (0, $\infty$), $L_i$ 의 값은 음수가 나오지 않도록 설계되었음.
- $W$ 학습의 첫 iteration에서 $s\approx0$ 인 상황이 벌어지면 loss는? $\rightarrow$ loss = C-1, $\Delta=1$ 이라는 가정하에 이게 성립함. Debugging 시 유용한 방법.
- Multi SVM loss의 경우 정답 카테고리와 같은 카테고리를 계산하지 않지만, 만약 계산한다면? $\rightarrow$ $\Delta$ 만큼만 커지게 된다. 
- Summation을 하지 않고 mean을 사용하게 되면? $\rightarrow$ C-1(constant) 만큼의 scaling을 한 값으로 나올뿐 크게 변하지 않음.
- max의 sqaured version?(L2-SVM) $\rightarrow$ 이는 결과에 대해서 큰 차이를 불러온다고 함. Loss가 큰 경우에 대해 Penalty를 더 부여하는 효과를 가져와 차이가 있음. 이는 데이터에 따라 결정해야하는 부분임. CV 과정에서 판단할 수 있다고 함.
- 일반적으로 max(0, ?) 꼴의 경우 **hinge loss** 라는 말로 불리기도 함.
- 아래는 Multiclass SVM loss 의 구현버젼.

```
def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in xrange(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i
  

def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i
```
___
### Regularization
- 그렇다면 이제 생각해야할 문제는 이렇게 구한 $W$가 unique 하냐의 문제. SVM loss는 합을 사용하기 떄문에 $W$에 상수만큼 곱해주면 똑같은 loss(=0)가 계산됨.
- 이러한 문제를 해결하기 위해 도입한 것이 바로 정규화(Regularization *회귀분석 Lasso, Ridge, Elastic Net*..) 
- 정규화를 간단히 말하면, 우리가 구할 parameter에 대해서 penalty를 부여해서 간단한 모델로 만들어주겠다는 idea. 성능 향상과 overfitting 방지에도 도움이 됨.
- formula
$$L(W)\text{ (full loss)} = \frac{1}{N}\sum_{i=1}^N L_i(f(x_i, W), y_i) \text{(data loss)} + \lambda R(W) \text{ (regularization loss)}$$
- Regularization 종류가 많음. 머신러닝, 딥러닝에도 많이 적용 됨.<br>
1) L2 Regularization : (L2 norm) $\sum_k\sum_lW^2_{k,l} or \sum_k\sum_l\sqrt{W_{k,l}} $
2) L1 Regularization : (L1 norm) $\sum_k\sum_l|{W_{k,l}}| \rightarrow \text{encouraging sparsity}$. 파라미터가 적은 방향을 지향함. 통계학에서 Lasso의 경우 가설검정을 동반한다고 표현.
- 여기서 첨언하자면, L2와 L1의 형태에 대해서 생각해보면 이유를 알 수 있음. L2는 원형꼴을 따르며, L1은 마름모 꼴을 따르게 됨. 따라서 만약 마름모의 꼭지점에 걸려버리면, 그 지점은 일부 파라미터가 0 값이 지정됨. 이러한 이유로 sparsity를 지원하는듯. <br>
cf) Elastic Net(L1+L2) /  Max Norm Regularization / Dropout - DL에 많이 사용 / Batch Normalization, Stochastic depth - DL에 많이 사용

___
### hyper-parameter?
- 이제 현실적으로 우리가 지정해줘야할 hyper-parameter가 남았음. $\Delta$와 $\lambda$.
- formula를 보면 loss는 data loss 와 regularization loss로 나뉘어서 tradeoff 관계가 성립하여 서로 영향을 줄 것으로 보이나,
- $\Delta$의 경우를 보면 loss만 커지거나 작아지지 실제로는 의미가 없는 hyper-parmeter임. $W$는 상당히 임의적으로 움직이는 parameter이기 때문. (디버깅의 이점으로 1이 선호될듯?)
- 실제로는 $\lambda$를 통해서 $W$에 제한을 거는 것이 영향력이 더 큼. 따라서 $\lambda$를 통해서 조절하는 것이 오히려 현실적임.
- 다만 L2 Regularization을 통해 SVM의 max margin 개념이 나왔다고 함... (나는 모르겠음 총총총..)
___
### Additional Consideration
- the # of classes = 2? $\rightarrow$ Binary SVM
- Loss function : $L_i=Cmax(0,1−y_iw^Tx_i)+R(W)$ $(-1<y_i<1)$
- $C$ 역시 hyper-parameter인데, multiclass에서의 $\lambda$와 비슷함. 관계는 $C\propto1/\lambda$.
- 기존 SVM과 관계가 많다고 하는데... 제가 SVM를 몰라서 ㅈㅅ합니다. 다시 찾아올게요.

___
### Logit? Sigmoid? Softmax?
- 얘네는 비슷하게 생겨서 관련이 있을까 싶지만 실제로 관계 정도가 아니라 똑같은 놈들임. 
- logit의 역함수 version : sigmoid
- sigmoid의 다변량 version : softmax (유도는 logit으로 부터)
- logit : $log(\frac{y}{1-y}) = f(x)$
- sigmoid : $y=\frac{e^{f(x)}}{1+e^{f(x)}}$ (activation function...)
- by multiclass logit $\rightarrow$ softmax: 
$$\frac{Pr(y_i|x)}{Pr(y_K|x)} = e^{f_{y_i}(x;W)} $$
$$\sum_{i \neq K}\frac{Pr(y_i|x)}{Pr(y_K|x)} = \sum_{i \neq K}e^{f_{y_i}(x;W)} $$
$$\frac{1-Pr(y_K|x)}{Pr(y_K|x)} =  \sum_{i \neq K}e^{f_{y_i}(x;W)}$$
$$ Pr(y_K|x) = \frac{1}{1+\sum_{i \neq K}e^{f_{y_i}(x;W)}}$$
$$ Pr(y_i|x) = e^{f_{y_i}(x;W)}Pr(y_K|x)$$
$$ Pr(y_i|x) = e^{f_{y_i}(x;W)}\frac{1}{1+\sum_{i \neq K}e^{f_{y_i}(x;W)}}$$
$$ Pr(y_i|x) = \frac{e^{f_{y_i}(x;W)}}{\sum e^{f_{y_i}(x;W)}}$$

___
### Softmax Classifier(Multinomial Logistic Regression) 
- **이제부터 확률로 생각합니다 만세~**
- 먼저, 여기서 기본적인 생각은 Score = unnormalized log probability of the classes.
- softmax에 score를 적용하면 Loss Function이 만들어짐.
- formula
$$ L_i = -logPr(y=k|X=x_i) = -log\frac{e^{s_k}}{\sum_j e^{s_j}}$$
- 왜 -log? 확률의 특성을 생각해서 1에 가까워지면 Loss가 최소가 되도록 Tuning한 것 <br>
- e.g. 고양이, 개, 개구리의 score가 $[3.2, 5.1, -1.7]$ 이라고하면 <br>
$[3.2, 5.1, -1.7]\rightarrow exp[24.5, 164.0, 0.18]\rightarrow normal[0.13, 0.87, 0.00]\rightarrowL_i = -log(0.13) = 0.89 $
- softmax도 몇가지 생각해볼 거리가 있음. <br>
1) loss의 가능한 영역은? $\rightarrow (0, \infty) $ 확률의 [0,1] 영역과 -log의 특성을 보면 알 수 있음. <br>
2) 학습의 첫 iteration에서 $s\approx 0 \text {인 상황이 벌어지면 loss는? }\rightarrow logC$, 역시 Debugging에 효율적임 <br>
3) softmax 는 Full Cross Entropy 라고도 함. 왜냐? <br>
- Information Theory에서 Cross Entropy는 true 분포의 p(x) 와 예측된 q(x) 간의 관계를 이용해 만든 식임. <br>
    $$ H(p,q) = -\sum_x p(x)logq(x) $$  <br>
- 여기서 p(x)는 정답 카테고리이기 떄문에 1이 반환되고, q(x)는 우리가 예측한 확률이 지정되는 형태를 갖게 되어 정확히 형태가 일치하게 됨 <br>
- True 집단과 Estimated 집단의 분포를을 비교하는 KL divergence적인 관점에서도 바라볼 수 있음. 다시 말해, 정답 카테고리의 분포에 최대한 비슷하게 갖다 붙여보겠다는 idea를 가진 것. <br>
- 또한 Cross Entropy는 Statistics 에서의 MLE 방법과 정확히 일치함. negative log-likelihood를 minimize하는 것이므로 MLE와 정확히 일치함.  <br>
- 계산적인 측면에서 컴퓨팅 문제가 있어 실제 구현할때는 상수 C를 위아래로 곱해줌. <br>

___
### Multiclass SVM loss & Softmax loss
- 먼저 각각의 Loss 결과를 통해 얻은 스코어, 확률은 각각의 방법 안에서만 비교하는 것이 의미가 있음.
- Softmax의 경우 결과로 확률을 제공한다는 점에서 신뢰성이 높다고 생각할 수 있다. 하지만, 이는 Regularization의 영향이 강해서 직접적으로 해석하는데 문제가 있다. $\lambda$를 통해 강한 제약을 건다면, 확률이 유니폼 형태에 가까워지므로 곧이 곧대로 해석할 수 없다.
- SVM과 Softmax를 비교하면, <br>
1) SVM의 경우 loss가 최소화되는 margin을 만족하면 거기서 더이상 개선하려고 하지 않음. <br>
2) 하지만 Softmax의 경우는 계속 더 나은 결과를 찾기위해 지속적으로 계산을 수행함.    

---
## How? Optimization!
- Loss function을 이제 알았다면 이제 그것을 최소화 시키는 방법이 필요하다. 
- 최적화를 하는 방법에 대해서는 Gradient를 계산하는 것이 가장 중요
