(ns deep-book.core
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray])
  (:require [org.apache.clojure-mxnet.random :as random])
  (:require [org.apache.clojure-mxnet.io :as mx-io])
  (:require [deep-book.mnist-loader :refer :all])
  (:gen-class))

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

(defn sigmoid_prime
  "Derivative of the sigmoid function."
  ([z]
   (let [sigmoid_z (sigmoid z)]
   (ndarray/* sigmoid_z (ndarray/- 1 sigmoid_z)))))




;def feedforward(self, a):
;        """Return the output of the network if "a" is input."""
;        for b, w in zip(self.biases, self.weights):
;            a = sigmoid(np.dot(w, a)+b)
;        return a


(defn feedforward ([^Network n
                    ^org.apache.mxnet.NDArray arr] ; is it axis?
                   "Return the output of the network if a is input."
                   (let [args (map vector (:biases n) (:weights n))]
                     (reduce (fn [a [b w]]
                               (sigmoid (-> (ndarray/dot w a)
                                            (ndarray/+ b)))) 
                             arr args))))


;def backprop(self, x, y):
;        """Return a tuple ``(nabla_b, nabla_w)`` representing the
;        gradient for the cost function C_x.  ``nabla_b`` and
;        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
;        to ``self.biases`` and ``self.weights``."""
;        nabla_b = [np.zeros(b.shape) for b in self.biases]
;        nabla_w = [np.zeros(w.shape) for w in self.weights]
;        # feedforward
;        activation = x
;        activations = [x] # list to store all the activations, layer by layer
;        zs = [] # list to store all the z vectors, layer by layer
;        for b, w in zip(self.biases, self.weights):
;            z = np.dot(w, activation)+b
;            zs.append(z)
;            activation = sigmoid(z)
;            activations.append(activation)
;        # backward pass
;        delta = self.cost_derivative(activations[-1], y) * \
;            sigmoid_prime(zs[-1])
;        nabla_b[-1] = delta
;        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
;        # Note that the variable l in the loop below is used a little
;        # differently to the notation in Chapter 2 of the book.  Here,
;        # l = 1 means the last layer of neurons, l = 2 is the
;        # second-last layer, and so on.  It's a renumbering of the
;        # scheme in the book, used here to take advantage of the fact
;        # that Python can use negative indices in lists.
;        for l in xrange(2, self.num_layers):
;            z = zs[-l]
;            sp = sigmoid_prime(z)
;            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
;            nabla_b[-l] = delta
;            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
;        return (nabla_b, nabla_w)


(defn copy0 
  ([nda]
  (mapv #(ndarray/zeros (ndarray/shape-vec %)) nda)))

(defn cost_derivative 
  ([output_activations y]
   (ndarray/- output_activations y)))


; returns vector with 2 elements [delta_nabla_b delta_nabla_w]
(defn backprop
  ([^Network n x y]
   (let [nabla_b (copy0 (:biases n))  ; seems we don't need this initialization!!!
         nabla_w (copy0 (:weights n))
;https://stackoverflow.com/questions/30828610/clojure-transform-list-of-pairs-n-tuples-into-n-tuple-of-lists
; Also, note that first element of za will be nil, which is not what we want!
         ; feedforward       
         [zs activations] (apply mapv vector 
                                 (reductions (fn [[z activation] [b w]]
                                               (let [step-z (-> (ndarray/dot w activation)
                                                                (ndarray/+ b))]
                                                 [step-z (sigmoid step-z)]))
                                             [nil x]
                                             (map vector (:biases n) (:weights n))) 
                                 )
         zs (subvec zs 1)
         ; backward pass
         delta (ndarray/* 
                (cost_derivative ((peek activations) y)) 
                (sigmoid_prime (peek zs)))
         nabla_b_last delta
         nabla_w_last (ndarray/dot delta (ndarray/transpose (nth activations (- (count activations) 2))))
         [nabla_b nabla_w] (apply mapv vector 
                                 (reductions (fn [[nb nw] [z w a]]
                                               (let [sp (sigmoid_prime z)
                                                     delta (ndarray/* (ndarray/dot (ndarray/transpose w) nb) sp)
                                                     ret_nw (ndarray/dot delta (ndarray/transpose a))]
                                                 [delta ret_nw]))
                                             [nabla_b_last nabla_w_last]
                                             (map vector 
                                                  (nthrest (reverse zs) 1) 
                                                  (reverse (nthrest (:weights n) 1))
                                                  (nthrest (reverse activations) 1))))
        ]
     [nabla_b nabla_w])))


;(mapv #(vector %1 %2) ["Ford" "Arthur" "Tricia"] ["a" "b" "c"])
;(apply mapv vector [[1 "a"], [2 "b"], [3 "c"]])
;(mapv vector [1 "a"], [2 "b"], [3 "c"])

;(let [x 10
;      [zs activations] (apply mapv vector 
;                              (reductions (fn [[z activation] [b w]]
;                                            (let [step-z (-> (* w activation)
;                                                             
;                                                             (+ b))]
;                                              [step-z (+ step-z step-z)]))
;                                          [nil x]
;                                          (map vector [1 2 3 4] [10 11 12 13])))
;      zs (subvec zs 1)]
;  zs)

; NOTE: returns NEW network with updated wights and biases !!!
(defn update-mini-batch 
  ([^Network n mini_batch eta]
   (let [eta_by_minibatch_cnt (/ eta (count mini_batch))
         step (fn [w nw]
                (->> (ndarray/* nw eta_by_minibatch_cnt)
                     (ndarray/- w)))
         nabla_b (copy0 (:biases n))
         nabla_w (copy0 (:weights n))
         [r_b r_w] (reduce (fn [[b w] [x y]] 
                             (let [[delta_nabla_b delta_nabla_w] (backprop n x y)] 
                               [(map ndarray/+ b delta_nabla_b) (map ndarray/+ w delta_nabla_w)]))
                           [nabla_b nabla_w]
                           mini_batch)]
     (assoc n 
            :weights (mapv step (:weights n) r_w)
            :biases (mapv step (:biases n) r_b)))
     
     ))

;(ndarray/->vec (ndarray/* (sigmoid (ndarray/ones [2 2])) 2.0))

(defn evaluate-network
  ([^Network n test_data]
   (let [test_results (map (fn [[x y]] [(ndarray/argmax (feedforward n x)) y]) test_data)]
     (count (filter (fn [[x y]] (= x y)) test_results)))))


;   def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
(defn SGD 
  ([^Network n training_data epochs mini_batch_size eta] 
   (SGD n training_data epochs mini_batch_size eta nil))
  
  ([^Network n training_data epochs mini_batch_size eta test_data]
   (let [n_test (count test_data)
         n (count training_data)
         ]
     (dotimes [j epochs] 
       (let [mini_batches (partition mini_batch_size (shuffle training_data))]
         (map #(update-mini-batch n % eta) mini_batches)
         (if (> n_test 0)
           (printf "Epoch %d: %s / %d" j (evaluate-network n test_data) n_test)
           (printf "Epoch %d complete" j))
         ))))
  )



(defn -main
  "First chapter of the book"
  [& args]
  (println "Starting training...")

  (let [net (make-network [784 30 10])
        ;train-data (mx-io/batch-data (mx-io/next train-data))
        ;test-data (mx-io/batch-data (mx-io/next test-data))
        ]
    ;(SGD net train-data 30 10 3.0 test-data))
    (vectorized-result 5)
    ))

;(-main)