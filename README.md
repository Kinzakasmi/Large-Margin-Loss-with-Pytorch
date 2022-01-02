# Large Margin Deep Networks for Classification with Pytorch
This is a Pytorch implementation of Large Margin Deep Networks for Classification defined in the following paper : https://arxiv.org/abs/1803.05598

The loss function is defined as following :

Consider a classification problem with n classes.  
Suppose we use a function <img src="https://render.githubusercontent.com/render/math?math=f_i: X \rightarrow R">, for i = 1,. . ., n that generates a prediction score for classifying the input vector x in X to class i.
Let (x_k, y_k) be the training set.  
Let h_l denote the output of the l’th layer (h_0 = x) and γ_l be the margin enforced for its corresponding representation. Then the margin loss can be defined as following :

<img src="https://render.githubusercontent.com/render/math?math=\hat{w} =  \argmin_w \sum_{l,k} A_{i \ne y_k} \max \left\{0, \gamma_l + \frac{f_i(x_k) - f_{y_k} (x_k)}{\| \bigtriangledown_{h_l}f_i(x_k) - \bigtriangledown_{h_l}f_{y_k}(x_k) \|_q} \right\}">
