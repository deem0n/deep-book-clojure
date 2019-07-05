(ns deep-book.ch1-test
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray])
  (:require [org.apache.clojure-mxnet.random :as random])
  (:require [clojure.test :refer :all]
            [deep-book.ch1 :refer :all]))

(deftest network-feedforward-test
  (let [out-layer-neuron-count 1024000
        net (make-network [10 1024 out-layer-neuron-count])
        input (ndarray/array [1 4 3 6 2 11 3 22 9 -12] [10 1])
        ff (feedforward net input)
        ]  
    (testing "FIXME, I fail."
      (is (= (ndarray/shape-vec ff) [out-layer-neuron-count 1])))))
