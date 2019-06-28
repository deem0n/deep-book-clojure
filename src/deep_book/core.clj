(ns deep-book.core
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray])
  (:require [org.apache.clojure-mxnet.random :as random])
  (:gen-class))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))

; https://stackoverflow.com/questions/11958027/clojures-defrecord-how-to-use-it
(defrecord Network [^java.lang.Long num_layers
                    ^clojure.lang.PersistentVector sizes ; number of neurons in the respective layers
                    ^clojure.lang.LazySeq biases
                    ^clojure.lang.LazySeq weights])
;The biases and weights in the Network object are all initialized randomly, using the Numpy np.random.randn function to generate Gaussian distributions with mean 0 and standard deviation 1.

;Constructor
(defn make-network ([sizes]
                    (->Network
                     (count sizes)
                     sizes
                     (map #(random/normal 0 1 [% 1]) (subvec sizes 1))
                     (map #(random/normal 0 1 [%2 %1]) (butlast sizes) (subvec sizes 1)))))

;(make-network [1923 256 10])



; I have no idea how to divide 1 by ndarray element-wise :-()
; ndarray/elemwise-div and ndarray/div both can not accept numbers as arguments
; so I am making ndarray with ones...

(defn sigmoid ([^org.apache.mxnet.NDArray z]
               "elementwise (in vectorized form)"
               (let [dyn-ones (ndarray/ones  (ndarray/shape-vec z))]
                 (ndarray/div dyn-ones
                              (-> (ndarray/* z -1.0)
                                  (ndarray/exp)
                                  (ndarray/+ 1.0))))))

;(ndarray/->vec (sigmoid (ndarray/ones [2 2])))

;def feedforward(self, a):
;        """Return the output of the network if "a" is input."""
;        for b, w in zip(self.biases, self.weights):
;            a = sigmoid(np.dot(w, a)+b)
;        return a


(defn feedforward ([^Network n
                    ^org.apache.mxnet.NDArray arr]
                   "Return the output of the network if a is input."
                   (let [args (map vector (:biases n) (:weights n))]
                     (reduce (fn [a [b w]]
                               (sigmoid (-> (ndarray/dot w a)
                                            (ndarray/+ b)))) arr args))))