(ns deep-book.mnist-loader-test
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray])
  (:require [clojure.test :refer :all]
            [deep-book.mnist-loader :refer :all]))



(deftest func-type 
  (testing "oops")
  (is (= "class org.apache.mxnet.NDArray" (str (type (vectorized-result 3))))))



(deftest ndarray-10-test
    (testing "FIXME, I fail."
      (is (= [0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0] 
             (ndarray/->vec (vectorized-result 3))))
      
      (is (= [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
             (ndarray/->vec (vectorized-result 0))))
      
      (is (= [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]
             (ndarray/->vec (vectorized-result 9))))))


(deftest returns-3-arrays
  (testing "load-data-wrapper is broken..."
    (let [array3 (load-data-wrapper)
          ]
      (is (instance? clojure.lang.PersistentVector array3))
      (is (= 50000 (count (first array3))))
      (is (= 10000 (count (second array3))))
      (is (= 10000 (count (nth array3 2))))
      (is (instance? clojure.lang.PersistentVector (first (nth array3 0))))
      (is (instance? org.apache.mxnet.NDArray  (first (first (nth array3 0)))))
      (is (instance? org.apache.mxnet.NDArray  (second (first (nth array3 0)))))
      (is (= [1 784] (ndarray/shape-vec (first (first (nth array3 0))))))
      (is (= [10 1] (ndarray/shape-vec (second (first (nth array3 0))))))
      
      
      (is (instance? org.apache.mxnet.NDArray  (first (first (nth array3 1)))))
      (is (instance? org.apache.mxnet.NDArray  (second (first (nth array3 1)))))
      (is (= [1 784] (ndarray/shape-vec (first (first (nth array3 1))))))
      (is (= [10 1] (ndarray/shape-vec (second (first (nth array3 1))))))  
      
      (is (instance? org.apache.mxnet.NDArray  (first (first (nth array3 2)))))
      (is (instance? org.apache.mxnet.NDArray  (second (first (nth array3 2)))))
      (is (= [1 784] (ndarray/shape-vec (first (first (nth array3 2))))))
      (is (= [10 1] (ndarray/shape-vec (second (first (nth array3 2))))))      
      )))


