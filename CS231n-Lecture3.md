# Lecture 3
___
### Lecture 2 에 이어서.
- Linear Classification을 통해 Score function을 생성함.
- score를 계산하는 fucntion의 parameter <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/>(weight)를 어떻게 찾을까? by Loss function, how? optimization!

___
## Loss function
- 경우에 따라 Cost fucntion으로 불리기도 함.
- loss(손실)을 최소화하면 되는거 아니냐의 idea
- 회귀에서도 error의 추정량의 제곱을 최소화 시키는 방법을 쓰듯..
- Loss over datasets 
<p align="center"><img src="/tex/ed664a1f56d03d5de2a46020ce85312f.svg?invert_in_darkmode&sanitize=true" align=middle width=194.14203104999999pt height=41.10931275pt/></p>

___
### Multiclass Support Vector Machine loss (Multiclass SVM loss)
- 이 손실함수의 idea는 어느 정도 허용수준(<img src="/tex/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode&sanitize=true" align=middle width=13.69867124999999pt height=22.465723500000017pt/>)이상으로 정답 클래스의 score를 더 크게 하는 방향으로 설계한다는 것.(good방향)
	- <img src="/tex/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode&sanitize=true" align=middle width=13.69867124999999pt height=22.465723500000017pt/> 는 hyper-parameter로 정해줘야하는 부분이다.
- formula
<p align="center"><img src="/tex/3010a3b8e3c3a4180737502486240c0d.svg?invert_in_darkmode&sanitize=true" align=middle width=217.20603464999996pt height=39.26959575pt/></p>
- 만약 계산된 3개의 사진에서 고양이, 개, 개구리인지 판단하는 스코어가 각각 <img src="/tex/65cafcffa0765f555e3f314b94197bf6.svg?invert_in_darkmode&sanitize=true" align=middle width=138.9155922pt height=24.65753399999998pt/> <img src="/tex/b3335b1b5f6bc000c6b11db1f99b9605.svg?invert_in_darkmode&sanitize=true" align=middle width=126.13015799999997pt height=24.65753399999998pt/> <img src="/tex/2439867047b3d6df03ee8e82a3608071.svg?invert_in_darkmode&sanitize=true" align=middle width=138.9155922pt height=24.65753399999998pt/> <img src="/tex/74c5d8b0bed1bc87f919cbc158fde18a.svg?invert_in_darkmode&sanitize=true" align=middle width=43.83551204999999pt height=22.465723500000017pt/> 이라고 하면, 
 <img src="/tex/64f62162627e2b7efb39973f1fa2a6ee.svg?invert_in_darkmode&sanitize=true" align=middle width=418.75552289999996pt height=24.65753399999998pt/> 
 <img src="/tex/f77dd5b437cfb0fd0e0863140c411a77.svg?invert_in_darkmode&sanitize=true" align=middle width=393.18465614999997pt height=24.65753399999998pt/> 
 <img src="/tex/5e47b9d64c7eae55c0a4211eaa217ad4.svg?invert_in_darkmode&sanitize=true" align=middle width=414.18929805pt height=24.65753399999998pt/>  
 <img src="/tex/cc1d135577a56717f43561ba7cce2370.svg?invert_in_darkmode&sanitize=true" align=middle width=212.10029445pt height=24.65753399999998pt/> 
 
- Multi-SVM loss에 대해서 생각해야할 것이 몇가지 있는데, formula에 근거해서 생각하면 됨.
- 만약 정답 카테고리의 score가 기존의 스코어의 순서를 바꾸지 않는 한에서 변하면 loss의 변화는? <img src="/tex/e5d134f35dc4949fab12ec64d186248a.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> loss는 변하지 않는다.
- Multi SVM loss의 가능한 값의 영역은? <img src="/tex/e5d134f35dc4949fab12ec64d186248a.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> (0, <img src="/tex/f7a0f24dc1f54ce82fecccbbf48fca93.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/>), <img src="/tex/6af2b4e795d7f62666e31c283eb02410.svg?invert_in_darkmode&sanitize=true" align=middle width=15.838142099999992pt height=22.465723500000017pt/> 의 값은 음수가 나오지 않도록 설계되었음.
- <img src="/tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/> 학습의 첫 iteration에서 <img src="/tex/2e4649eb610a632b8a13192a7ac57caf.svg?invert_in_darkmode&sanitize=true" align=middle width=37.84231934999999pt height=21.18721440000001pt/> 인 상황이 벌어지면 loss는? <img src="/tex/e5d134f35dc4949fab12ec64d186248a.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> loss = C-1, <img src="/tex/e70a40381a56333dfaa505fb450d143a.svg?invert_in_darkmode&sanitize=true" align=middle width=43.83551204999999pt height=22.465723500000017pt/> 이라는 가정하에 이게 성립함. Debugging 시 유용한 방법.
- Multi SVM loss의 경우 정답 카테고리와 같은 카테고리를 계산하지 않지만, 만약 계산한다면? <img src="/tex/e5d134f35dc4949fab12ec64d186248a.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> <img src="/tex/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode&sanitize=true" align=middle width=13.69867124999999pt height=22.465723500000017pt/> 만큼만 커지게 된다. 
- Summation을 하지 않고 mean을 사용하게 되면? <img src="/tex/e5d134f35dc4949fab12ec64d186248a.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> C-1(constant) 만큼의 scaling을 한 값으로 나올뿐 크게 변하지 않음.
- max의 sqaured version?(L2-SVM) <img src="/tex/e5d134f35dc4949fab12ec64d186248a.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> 이는 결과에 대해서 큰 차이를 불러온다고 함. Loss가 큰 경우에 대해 Penalty를 더 부여하는 효과를 가져와 차이가 있음. 이는 데이터에 따라 결정해야하는 부분임. CV 과정에서 판단할 수 있다고 함.
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
- 그렇다면 이제 생각해야할 문제는 이렇게 구한 <img src="/tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/>가 unique 하냐의 문제. SVM loss는 합을 사용하기 떄문에 <img src="/tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/>에 상수만큼 곱해주면 똑같은 loss(=0)가 계산됨.
- 이러한 문제를 해결하기 위해 도입한 것이 바로 정규화(Regularization *회귀분석 Lasso, Ridge, Elastic Net*..) 
- 정규화를 간단히 말하면, 우리가 구할 parameter에 대해서 penalty를 부여해서 간단한 모델로 만들어주겠다는 idea. 성능 향상과 overfitting 방지에도 도움이 됨.
- formula
<p align="center"><img src="/tex/745668059c3e045c7f46c21ba95f02d7.svg?invert_in_darkmode&sanitize=true" align=middle width=594.60520515pt height=47.806078649999996pt/></p>
- Regularization 종류가 많음. 머신러닝, 딥러닝에도 많이 적용 됨.<br>
1) L2 Regularization : (L2 norm) <img src="/tex/e4aaf9a782a3f3f9583c9506a1bf8296.svg?invert_in_darkmode&sanitize=true" align=middle width=205.13343014999998pt height=27.73529880000001pt/>
2) L1 Regularization : (L1 norm) <img src="/tex/e57c9c48ebe23f1473f09b2c3168d566.svg?invert_in_darkmode&sanitize=true" align=middle width=267.29493285pt height=24.657735299999988pt/>. 파라미터가 적은 방향을 지향함. 통계학에서 Lasso의 경우 가설검정을 동반한다고 표현.
- 여기서 첨언하자면, L2와 L1의 형태에 대해서 생각해보면 이유를 알 수 있음. L2는 원형꼴을 따르며, L1은 마름모 꼴을 따르게 됨. 따라서 만약 마름모의 꼭지점에 걸려버리면, 그 지점은 일부 파라미터가 0 값이 지정됨. 이러한 이유로 sparsity를 지원하는듯. <br>
cf) Elastic Net(L1+L2) /  Max Norm Regularization / Dropout - DL에 많이 사용 / Batch Normalization, Stochastic depth - DL에 많이 사용

___
### hyper-parameter?
- 이제 현실적으로 우리가 지정해줘야할 hyper-parameter가 남았음. <img src="/tex/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode&sanitize=true" align=middle width=13.69867124999999pt height=22.465723500000017pt/>와 <img src="/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/>.
- formula를 보면 loss는 data loss 와 regularization loss로 나뉘어서 tradeoff 관계가 성립하여 서로 영향을 줄 것으로 보이나,
- <img src="/tex/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode&sanitize=true" align=middle width=13.69867124999999pt height=22.465723500000017pt/>의 경우를 보면 loss만 커지거나 작아지지 실제로는 의미가 없는 hyper-parmeter임. <img src="/tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/>는 상당히 임의적으로 움직이는 parameter이기 때문. (디버깅의 이점으로 1이 선호될듯?)
- 실제로는 <img src="/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/>를 통해서 <img src="/tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/>에 제한을 거는 것이 영향력이 더 큼. 따라서 <img src="/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/>를 통해서 조절하는 것이 오히려 현실적임.
- 다만 L2 Regularization을 통해 SVM의 max margin 개념이 나왔다고 함... (나는 모르겠음 총총총..)
___
### Additional Consideration
- the # of classes = 2? <img src="/tex/e5d134f35dc4949fab12ec64d186248a.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> Binary SVM
- Loss function : <img src="/tex/7b4e2b3e8e2d0aec570fb28830b24dea.svg?invert_in_darkmode&sanitize=true" align=middle width=234.80908935pt height=27.6567522pt/> <img src="/tex/abbda87fbc1c49b5a02accc53bc582da.svg?invert_in_darkmode&sanitize=true" align=middle width=99.37677419999999pt height=24.65753399999998pt/>
- <img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/> 역시 hyper-parameter인데, multiclass에서의 <img src="/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/>와 비슷함. 관계는 <img src="/tex/7de40c4f975c63ff4e83e093d8388d52.svg?invert_in_darkmode&sanitize=true" align=middle width=60.86977049999999pt height=24.65753399999998pt/>.
- 기존 SVM과 관계가 많다고 하는데... 제가 SVM를 몰라서 ㅈㅅ합니다. 다시 찾아올게요.

___
### Logit? Sigmoid? Softmax?
- 얘네는 비슷하게 생겨서 관련이 있을까 싶지만 실제로 관계 정도가 아니라 똑같은 놈들임. 
- logit의 역함수 version : sigmoid
- sigmoid의 다변량 version : softmax (유도는 logit으로 부터)
- logit : <img src="/tex/53f2f0c259034a48e257422c774725c0.svg?invert_in_darkmode&sanitize=true" align=middle width=116.17897169999999pt height=24.65753399999998pt/>
- sigmoid : <img src="/tex/7d982e6b3fa5c39de42703c01a456ee8.svg?invert_in_darkmode&sanitize=true" align=middle width=78.24577034999999pt height=35.19487620000001pt/> (activation function...)
- by multiclass logit <img src="/tex/e5d134f35dc4949fab12ec64d186248a.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> softmax: 
<p align="center"><img src="/tex/07670e8f6b4154c9c2b2d6c670049219.svg?invert_in_darkmode&sanitize=true" align=middle width=153.75993105pt height=38.83491479999999pt/></p>
<p align="center"><img src="/tex/a8dd8e134913956eb35eacad73dbb879.svg?invert_in_darkmode&sanitize=true" align=middle width=214.39967175pt height=45.4586385pt/></p>
<p align="center"><img src="/tex/be8441f9f94dd1a00dc465d441b7db9a.svg?invert_in_darkmode&sanitize=true" align=middle width=211.4038905pt height=45.4586385pt/></p>
<p align="center"><img src="/tex/abfa17d99d84adfb5be8e74e3aa69aad.svg?invert_in_darkmode&sanitize=true" align=middle width=230.39936699999998pt height=41.765103599999996pt/></p>
<p align="center"><img src="/tex/c98ac0c5ba438414b55ee1c87c77c3ab.svg?invert_in_darkmode&sanitize=true" align=middle width=213.59784434999997pt height=19.526994300000002pt/></p>
<p align="center"><img src="/tex/1633ef5a1036a1d0fa849b6a5e6b69ef.svg?invert_in_darkmode&sanitize=true" align=middle width=285.70080824999997pt height=41.765103599999996pt/></p>
<p align="center"><img src="/tex/9344b0dfd8445aaa07c66027332160a3.svg?invert_in_darkmode&sanitize=true" align=middle width=167.47236329999998pt height=42.8023266pt/></p>

___
### Softmax Classifier(Multinomial Logistic Regression) 
- **이제부터 확률로 생각합니다 만세~**
- 먼저, 여기서 기본적인 생각은 Score = unnormalized log probability of the classes.
- softmax에 score를 적용하면 Loss Function이 만들어짐.
- formula
<p align="center"><img src="/tex/1935475f7c6829beda884c09730dbb0b.svg?invert_in_darkmode&sanitize=true" align=middle width=308.52920009999997pt height=40.4852712pt/></p>
- 왜 -log? 확률의 특성을 생각해서 1에 가까워지면 Loss가 최소가 되도록 Tuning한 것 <br>
- e.g. 고양이, 개, 개구리의 score가 <img src="/tex/244a22f4138a85e7d0b13e4d69b5cda4.svg?invert_in_darkmode&sanitize=true" align=middle width=99.54357599999997pt height=24.65753399999998pt/> 이라고하면 <br>
<img src="/tex/1feff6d0120c46caea4a9b8606d7d0cc.svg?invert_in_darkmode&sanitize=true" align=middle width=616.0670851499999pt height=24.65753399999998pt/>
- softmax도 몇가지 생각해볼 거리가 있음. <br>
1) loss의 가능한 영역은? <img src="/tex/b501694192ee148eb4c21e5fcaf60c5a.svg?invert_in_darkmode&sanitize=true" align=middle width=65.75343059999999pt height=24.65753399999998pt/> 확률의 [0,1] 영역과 -log의 특성을 보면 알 수 있음. <br>
2) 학습의 첫 iteration에서 <img src="/tex/c92c3feaeb9e884ed24edce246350e72.svg?invert_in_darkmode&sanitize=true" align=middle width=155.224608pt height=22.831056599999986pt/>, 역시 Debugging에 효율적임 <br>
- softmax 는 Full Cross Entropy 라고도 함. 왜냐? <br>
- Information Theory에서 Cross Entropy는 true 분포의 p(x) 와 예측된 q(x) 간의 관계를 이용해 만든 식임. <br>
    $$ H(p,q) = -\sum_x p(x)logq(x) $$  <br>
- 여기서 p(x)는 정답 카테고리이기 떄문에 1이 반환되고, q(x)는 우리가 예측한 확률이 지정되는 형태를 갖게 되어 정확히 형태가 일치하게 됨 <br>
- True 집단과 Estimated 집단의 분포를을 비교하는 KL divergence적인 관점에서도 바라볼 수 있음. 다시 말해, 정답 카테고리의 분포에 최대한 비슷하게 갖다 붙여보겠다는 idea를 가진 것. <br>
- 또한 Cross Entropy는 Statistics 에서의 MLE 방법과 정확히 일치함. negative log-likelihood를 minimize하는 것이므로 MLE와 정확히 일치함.  <br>
- 계산적인 측면에서 컴퓨팅 문제가 있어 실제 구현할때는 상수 C를 위아래로 곱해줌. <br>

___
### Multiclass SVM loss & Softmax loss
- 먼저 각각의 Loss 결과를 통해 얻은 스코어, 확률은 각각의 방법 안에서만 비교하는 것이 의미가 있음.
- Softmax의 경우 결과로 확률을 제공한다는 점에서 신뢰성이 높다고 생각할 수 있다. 하지만, 이는 Regularization의 영향이 강해서 직접적으로 해석하는데 문제가 있다. <img src="/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/>를 통해 강한 제약을 건다면, 확률이 유니폼 형태에 가까워지므로 곧이 곧대로 해석할 수 없다.
- SVM과 Softmax를 비교하면, <br>
1) SVM의 경우 loss가 최소화되는 margin을 만족하면 거기서 더이상 개선하려고 하지 않음. <br>
2) 하지만 Softmax의 경우는 계속 더 나은 결과를 찾기위해 지속적으로 계산을 수행함.    

---
## How? Optimization!
- Loss function을 이제 알았다면 이제 그것을 최소화 시키는 방법이 필요하다. 
- 최적화를 하는 방법에 대해서는 Gradient를 계산하는 것이 가장 중요
