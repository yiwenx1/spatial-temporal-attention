$X_t = [x_{t,1}, x_{t,2}, ..., x_{t,i}, ..., x_{t,196}]$

$e_{t, i}^s = W_h^s \times h_{t-1} + W_x \times X_t + b$

$\alpha_{t, i} =\frac{exp(e_{t,i}^{s})}{\sum_{j=1}^{L^2}exp(e_{t, j}^{s})}$

$Y_t = \sum_{i=1}^{L^2}\alpha_{t, i}x_{t,i}$

$e_{t,i}^t = W_h^t \times h_{t-1} + W_y \times Y_t +b$

$\beta_{t}=\frac{exp(e_{t,i}^t)}{\sum_{j=1}^{L^2}exp(e_{t, j}^t)}$

$o = \sum_{t=1}^T \beta_t (W \times h_t)$

$\hat{z_{i}} = \frac{exp(o_i)}{\sum_{j=1}^{C}exp(o_j)}$

$L=-\sum_{i=1}^C z_i log\hat{z_i} + \lambda_1 \sum_{i=1}^{L^2}(1-\sum_{t=1}^T\alpha_{t,i}) + \lambda_2 \sum_{t=1}^T||\beta_t||$

