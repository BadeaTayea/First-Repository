# Classification: Predicting Categorical Data 
- **Usage:** Classification is concerned with predicting the classes, or labels, of a data sample based on its features.  
- **Example Applications:**
  - Recognizing hand-written digits
  - Marking Emails: Spam / Non-Spam 
  - Image Classification: Human / Animal
- **Types:**
  - ***Binary Classification***
  - ***Multi-Class Clasification***
  - Multi-Label Classification
  - Imbalanced Classification
 ## Binary Classification with SciKit Learn:
  - **Given:**
    - Two Class Labels (One class is assigned label:0; the other is assigned label:1)
  - **Popular Algorithms:**
    - Logistic Rergession 
    - k-Nearest Neigbors 
    - Decision Trees 
    - Support Vector Machines 
    - Naive Bayes
  - **Worked Example:** 
    - Training a simple "5-Detector" Binary Classifier:

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/classification/mnist_digit.png", height="120"/>
</p>

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/classification/mnist.png", height="240"/>  
</p>

<pre>
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
      
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
      
y_train_5 = (y_train == 5)  
y_test_5 = (y_test == 5)
      
# Creating a SGD Classifier and training it on the whole training set:
<b>from sklearn.linear_model import SGDClassifier</b>
      
# SGDClassifier relies on randomness during training, so setting the random_state parameter is important. 
<b>sgd_clf = SGDClassifier(random_state = 42)</b> 
<b>sgd_clf.fit(X_train, y_train_5)</b>
      
# Predicting:
# digit = X[0]
# <b>sgd_clf.predict([digit])</b> 
      
</pre>

## Performance Measures
### 1. Measuring Accuracy Using Cross Validation
  - **Notes:**
    - ***Cross Validation:*** A resampling procedure used to evaluate machine learning models on a limited data sample. As such, the procedure is often called ***k-fold cross-validation***.
    - This approach involves randomly dividing the set of observations into k groups, or folds, of approximately equal size. The first fold is treated as a validation set, and the method is fit on the remaining k−1 folds.
    - Accuracy is generally not the preferred performance measure for classifiers, especially when dealing with ***skewed datasets*** (i.e., when some classes are much more frequent than others).
  - **Code:**
<pre>
<b>from sklearn.model_selection import cross_val_score</b>
<b>cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")</b>
</pre>

---

### 2. Confusion Matrix
  - **Notes:**
    - ***Confusion matrix***: A technique for summarizing the performance of a classification algorithm.
    - In a two-class problem, we are often looking to discriminate between observations with a specific outcome from normal observations.
    - In this way, we can assign the event row as "positive" and the no-event row as "negative". We can then assign the event column of predictions as “true” and the no-event as "false".
      - TP: for correctly predicted event values.
      - FP: for incorrectly predicted event values.
      - TN: for correctly predicted no-event values.
      - FN: for incorrectly predicted no-event values.
    - This can help in calculating more advanced classification metrics such as ***precision***, ***recall***, ***specificity*** and ***sensitivity*** of our classifier.
    

<center>
  
|   | **Event**  | **No-Event**  |
|---|---|---|
| **Event**  | TP  | FP  |
| **No-Event**  | FN  | TN  |
  
</center>


  - **Code:**
<pre>
# Confusion Matrix:
<b>from sklearn.metrics import confusion_matrix</b>
<b>conf_mx = confusion_matrix(y_train_5, y_train_pred)</b>

# Precision & Recall of Classifier:
<b>from sklearn.metrics import precision_score, recall_score</b>
<b>precision_score(y_train_5, y_train_pred)</b>
<b>recall_score(y_train_5, y_train_pred)</b>
</pre>

---

### 3. Precision/Recall Tradeoff 
 - **Notes:**
   - Increasing precision reduces recall, and vice versa. This is called the ***precision/recall tradeoff***.
   - ***Precision*** quantifies the number of positive class predictions that actually belong to the positive class.
   - ***Recall*** quantifies the number of positive class predictions made out of all positive examples in the dataset.

 - **Code:**
<pre>
# Computing Precision, Recall, and Decision Threshold
<b>from sklearn.metrics import precision_recall_curve</b>
<b>precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)</b>

</pre>

<p align="center">
  <img src=https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/classification/precision_recall.png/>
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/classification/precision_recall2.png"/>
</p>

---

### 4. ROC Curves
- **Notes:**
  - ***Receiver operating characteristic (ROC) curve*** is another common tool used with binary classifiers. 
  - It is very similar to the precision/recall curve, but instead of plotting precision versus recall, the ROC curve plots the true positive rate (another name for recall) against the false positive rate.
    - The FPR is the ratio of negative instances that are incorrectly classified as positive. It is equal to one minus the true negative rate, which is the ratio of negative instances that are correctly classified as negative.
    - The TNR is also called ***specificity***.
    - Hence the ROC curve plots ***sensitivity (recall)*** versus 1 – specificity.
  - One way to compare classifiers is to measure the ***area under the curve (AUC)***. 
    - A perfect classifier will have a ROC AUC equal to 1
 
- **Code:**
<pre>
<b>from sklearn.metrics import roc_curve</b>
<b>fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)</b>
</pre>

<p align="center">
  <img src=https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/classification/roc_curve.png/>
</p>

---

# Optimization
- **Notes:**
  - ***Gradient Descent:*** is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems.
  - The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function.
  - Types of Gradient Descent Algorithm:
    - Batch Gradient Descent 
    - Stochastic Gradient Descent 
    - Mini-Batch Gradient Descent 
 
## Batch Gradient Descent 
  - **Notes:**
    - ***Batch Gradient Descent*** uses the whole batch of training data at every step. 
      - As a result, it is terribly slow on very large training sets. 
    - Learning rate affects the convergence to a minimum.
  - **Code:**
<pre>
# Generating "Noisy" Linear Data:
import numpy as np 
np.random.seed(21)
X = 2*np.random.randn(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)

<b># LinearRegression Using Batch Gradient:</b>
from sklearn.linear_model import LinearRegression</b>
lin_reg = LinearRegression()</b>
lin_reg.fit(X, y)</b>
lin_reg.intercept_, lin_reg.coef_
</pre>

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/optimization_linear_regression/linear_reg_bgd.png"/>
</p>

<p align="center">
  <img src=https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/optimization_linear_regression/bgd_alphas.png>
</p>

## Stochastic Gradient Descent 
- **Notes:**
  - ***Stochastic Gradient Descent*** just picks a random instance in the training set at every step and computes the gradients based only on that single instance, making it the faster algorithm.
  - It also makes it possible to train on huge training sets, since only one instance needs to be in memory at each iteration
  - On the other hand, due to its stochastic (i.e., random) nature, this algorithm is much less regular than Batch Gradient Descent: instead of gently decreasing until it reaches the minimum, the cost function will bounce up and down, decreasing only on average.
  - Stochastic Gradient Descent has a better chance of finding the global minimum than Batch Gradient Descent does.
  - Randomness is good to escape from local optima, but bad because it means that the algorithm can never settle at the minimum. One solution to this dilemma is to gradually reduce the learning rate.
    - The function that determines the learning rate at each iteration is called the ***learning schedule***.

- **Code:**
<pre>
# Using Sci-Kit Learn's LinearRegression Function:
<b>from sklearn.linear_model import SGDRegressor</b>
<b>sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty=None, eta0=0.1, random_state=42)</b>
<b>sgd_reg.fit(X, y.ravel())</b>
<b>sgd_reg.intercept_, sgd_reg.coef_</b>
</pre>

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/optimization_linear_regression/linear_reg_sgd.png"/>
</p>

## Mini-Batch Gradient Descent 
- **Notes:**
  - At each step, instead of computing the gradients based on the full training set (as in Batch GD) or based on just one instance (as in Stochastic GD), ***Minibatch GD*** computes the gradients on small random sets of instances called ***minibatches***.
  - The main advantage of Mini-batch GD over Stochastic GD is that you can get a performance boost from hardware optimization of matrix operations, especially when using GPUs.
  - Mini-batch GD will end up walking around a bit closer to the minimum than SGD. But, on the other hand, it may be harder for it to escape from local minima.

- **Code:**
<pre>
# Implementation
theta_path_mgd = []

n_iterations = 50
minibatch_size = 20

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        alpha = learning_schedule(t)
        theta = theta - alpha * gradients
        theta_path_mgd.append(theta)
</pre>

## Sum-Up
<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/optimization_linear_regression/gradients.png"/>
</p>

- All variants end up near the minimum, but Batch GD’s path actually stops at the minimum, while both Stochastic GD and Mini-batch GD continue to walk around.
- Batch GD takes a lot of time to take each step, and Stochastic GD and Mini-batch GD would also reach the minimum if a good learning schedule is used.


# Regression: Predicting Continuous Data
## Polynomial Regression
- **Notes:**
  -  If the data is actually more complex than a simple straight line, a linear model to fit nonlinear data could be used.
  -  A simple way to do this is to add powers of each feature as new features, then train a linear model on this extended set of features. This technique is called ***Polynomial Regression***.

- **Code:**
<pre>
<b>from sklearn.preprocessing import PolynomialFeatures</b>

<b>poly_features = PolynomialFeatures(degree=2, include_bias=False)</b>
<b>X_poly = poly_features.fit_transform(X)</b>
</pre>

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/regression/nonlinear_reg.png"/>
</p>

## Learning Curves
- **Notes:**
  -  If we perform high-degree Polynomial Regression, we will likely fit the training data much better than with plain Linear Regression.
  - The 300-degree polynomial model wiggles around to get as close as possible to the training instances.

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/regression/high_deg_pol_reg.png"/>
</p>

  -  The high-degree Polynomial Regression model is severely overfitting the training data, while the linear model is underfitting it.
  -  The model that generalizes seemingly best in this case is the quadratic model.
  -  It makes sense since the data was generated using a quadratic model, but in general we won’t know what function generated the data, so how can we decide how complex our model should be? How can we tell that our model is ***overfitting*** or ***underfitting*** the data?
  -  One way is to look at the ***learning curves***.

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/regression/learning_curves.png"/>
</p>

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/regression/learning_curves_2.png"/>
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/regression/learning_curves_3.png"/>

</p>

## Regularized Linear Models
- **Notes:**
  -  A good way to reduce overfitting is to ***regularize the model*** (i.e., to constrain it): the fewer degrees of freedom it has, the harder it will be for it to overfit the data.
  -  For a linear model, regularization is typically achieved by constraining the weights of the model.
  -  Three methods for constraining the weights:
     - **Ridge Regression**
     - **Lasso Regression**
     - **Elastic Net**

### Ridge Regression
- **Notes:**
  - ***Ridge Regression*** (also called ***Tikhonov regularization***) is a regularized version of Linear Regression: a regularization term is added to the cost function.
  - This forces the learning algorithm to not only fit the data but also keep the model weights as small as possible.
  - The regularization term should only be added to the cost function during training. Once the model is trained, the model’s performance is evaluated using the unregularized performance measure.
  - The ***hyperparameter  α***  controls how much the model be regularized.
    - If  α=0 , then Ridge Regression is just Linear Regression.
    - If  α  is very large, then all weights end up very close to zero and the result is a flat line going through the data’s mean.

- **Code:**
<pre>
from sklearn.linear_model import Ridge

# Noisy Linear Data:
np.random.seed(21)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)

# Regressor Function
def model(model_class, polynomial, alphas, **model_kargs):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        
# Specifying Ridge Regression:
<b>model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)</b>
</pre>

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/regression/ridge_reg.png"/>
</p>

- Note how increasing  α  leads to flatter (i.e., less extreme, more reasonable) predictions; this reduces the model’s variance but increases its bias.

---

### Lasso Regression
- **Notes:**
  - ***Least Absolute Shrinkage and Selection Operator Regression*** (simply called ***Lasso Regression***) is another regularized version of Linear Regression.
  - An important characteristic of Lasso Regression is that it tends to completely eliminate the weights of the least important features (i.e., set them to zero)

<pre>
<b>from sklearn.linear_model import Lasso</b>
<b>plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)</b>
<b>plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1, random_state=42)</b>
</pre>

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/regression/lasso_reg.png"/>
</p>

---

### Elastic Net
- **Notes:**
  - ***Elastic Net*** is a middle ground between Ridge Regression and Lasso Regression.
  - The regularization term is a simple mix of both Ridge and Lasso’s regularization terms, and mix ratio  r  can be controlled.
    - When  r=0 , Elastic Net is equivalent to Ridge Regression
    - When  r=1 , it is equivalent to Lasso Regression.

---

### Sum-Up
- So when should plain Linear Regression (i.e., without any regularization), Ridge, Lasso, or Elastic Net be used?
  - It is almost always preferable to have at least a little bit of regularization, so generally plain **Linear Regression** should be avoided.
  - **Ridge** is a good default, but if you suspect that only a few features are actually useful, you should prefer **Lasso** or **Elastic Net** since they tend to reduce the useless features’ weights down to zero as discussed.
In general, **Elastic Net** is preferred over Lasso since Lasso may behave erratically when the number of features is greater than the number of training instances or when several features are strongly correlated.

---

# Support Vector Machines
- **Notes:**
  - A ***Support Vector Machine (SVM)*** is a very powerful and versatile Machine Learning model, capable of performing linear or nonlinear classification, regression, and even outlier detection.
  - SVMs are particularly well suited for classification of complex but small- or medium-sized datasets.

- **Code:**
<pre>
<b>from sklearn.svm import SVC</b>
from sklearn import datasets

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# SVM Classifier model
<b>svm_clf = SVC(kernel="linear", C=float("inf"))</b>
<b>svm_clf.fit(X, y)</b>

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    <b>svs = svm_clf.support_vectors_</b>
</pre>

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/svms/iris_classification.png"/>
</p>

- An SVM classifier can be thought of as a model fitting the widest possible street (represented by the parallel dashed lines) between the classes. This is called ***large margin classification***.
- Note: Adding more training instances “off the street” will not affect the decision boundary at all: it is fully determined (or “supported”) by the instances located on the edge of the street. These instances are called the ***support vectors***.

## Hard Margin Classification
- **Notes:** 
  - If we strictly impose that all instances be off the street and on the right side, this is called ***hard margin classification***.
  - There are two main issues with hard margin classification:
    - It only works if the data is linearly separable.
    - It is quite sensitive to outliers.

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/svms/soft_margin_classification.png"/>
</p>

## Soft Margin Classification
- **Notes:**
  - The objective is to find a good balance between keeping the street as large as possible and limiting the margin violations (i.e., instances that end up in the middle of the street or even on the wrong side). This is called ***soft margin classification***.
  - In Scikit-Learn’s SVM classes, we can control this balance using the ***C-parameter***: a smaller C value leads to a wider street but more margin violations.

- **Code:**
<pre>
<b>
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris virginica

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])

svm_clf.fit(X, y)


scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

scaled_svm_clf1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ])
scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])

scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)

# Find support vectors (LinearSVC does not do this automatically)
t = y * 2 - 1
support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]
</b>
</pre>


## Nonlinear SVM Classification
- **Notes:**
  - Although linear SVM classifiers are efficient and work surprisingly well in many cases, many datasets are not even close to being linearly separable.
  - One approach to handling nonlinear datasets is to add more features, such as polynomial features; in some cases this can result in a linearly separable dataset

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/svms/nonlinear_svm.png"/>
</p>

- Application on the Moons Dataset:
- **Code:**
<pre>
<b>
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

polynomial_svm_clf.fit(X, y)
</b>
</pre>

<p align="center">
  <img src="https://github.com/BadeaTayea/First-Repository/blob/master/phys491_img/svms/moons_classification.png"/>
</p>

# SciKit Learn - Function/Class Documentation
- **Classification:**
- **SVM:**
- **Regression:**
- **Optimization:**

# Main Reference:
- Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems by Aurélien Géron 

