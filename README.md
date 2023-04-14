# Elastic_net
Elastic_net cython implementation
1. To compile the *.pyx make sure Cython is installed.
  1.1  pip install Cython
2. After that compile the elastic_net.pyx code by running:
  2.1 python setup.py build_ext --inplace
  
3. To use the ElasticNet class from the elastic_net code just add the following lines to your python script.
  import sys
  sys.path.append({your_path_to_the_elastic_net_directory_where_the_files_where_downloaded_and_compiled")
  from elastic_net import ElasticNet


ElasticNet(alpha=1, l1_ratio=0.5, max_iter=100, tol=0.0001)

Example:
#Your Data (X) and target_variables (y) should be np.arrays
#In this example the X,y have been split into training and testing data sets
# You can use the following function
4. def test_train_split(x_array, y_array, train_size = 0.6, shuffle=True):
      if x_array.ndim > 1:
          np.random.shuffle(x_array)
      new_matrix = np.append(x_array, y_array.reshape(-1, 1), axis=1)
      num_rows = int((np.shape(new_matrix)[0]) * train_size)
      x_train, y_train = new_matrix[:num_rows, :-1], new_matrix[:num_rows, -1]
      x_test, y_test = new_matrix[num_rows:, :-1], new_matrix[num_rows:, -1]
      return x_train, y_train, x_test, y_test
    
5. x_train, y_train, x_test, y_test = test_train_split(X,y)

6.model = ElasticNet(alpha=a_, l1_ratio=l1_iter, max_iter=max_iter, tol=0.0001)
 #######################################################################################################################
  Case 1
  l1_ratio = 0 ; Ridge Regression (l2 normalization) and alpha any number
  Case 2
  l1_ratio = 1 ; Lasso Regression (l1 normalization) and alpha any number
  Case 3
  l1_ratio = 0 and alpha =0; Linear regression
 ####################################################################################################################### 
  
 6.1 model = ElasticNet(alpha=0, l1_ratio=0.0, max_iter=1000, tol=0.0001)
     model.fit(x_train, y_train)
     y_pred, error = model.predict(x_test,y_test, intercept = False)
     error_ : Mean squared error of the y_test/y_predicted 
     model.coef_ : access the coefficients of the Beta vector
