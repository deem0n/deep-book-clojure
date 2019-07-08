(ns deep-book.ch1
  (:require [clojure.pprint])
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray])
  (:require [org.apache.clojure-mxnet.random :as random])
  (:require [org.apache.clojure-mxnet.io :as mx-io])
  (:require [deep-book.mnist-loader :refer :all])
  (:require [deep-book.mlflow :refer :all]))


; how to debug https://cambium.consulting/articles/2018/2/8/the-power-of-clojure-debugging



; https://stackoverflow.com/questions/11958027/clojures-defrecord-how-to-use-it
(defrecord Network [^java.lang.Long num_layers
                    ^clojure.lang.PersistentVector sizes ; number of neurons in the respective layers
                    ^clojure.lang.LazySeq biases
                    ^clojure.lang.LazySeq weights
                    ^org.mlflow.tracking.MlflowClient mlflow
                    ^String experiment-title
                    experiment-id
                    ^org.mlflow.api.proto.Service$RunInfo run-info
                    run-id])
;The biases and weights in the Network object are all initialized randomly, using the Numpy np.random.randn function to generate Gaussian distributions with mean 0 and standard deviation 1.

;Constructor
(defn make-network ([sizes mlflow-tracking-url]
                    (->Network
                     (count sizes)
                     sizes
                     (map #(random/normal 0 1 [% 1]) (subvec sizes 1))
                     (map #(random/normal 0 1 [%2 %1]) (butlast sizes) (subvec sizes 1))
                     (if mlflow-tracking-url
                       (new org.mlflow.tracking.MlflowClient mlflow-tracking-url)
                       nil)
                     nil
                     nil
                     nil
                     nil)))

;(make-network [1923 256 10])

(defn prepare-experiment
  ([^Network net experiment-title]
   (if (:mlflow net)
     (let [experiment-id (deep-book.mlflow/get-experiment-id (:mlflow net) experiment-title)
           run-info (.. (:mlflow net) (createRun experiment-id))
           run-id   (.. run-info (getRunId))
           ]
       (assoc net 
              :experiment-title experiment-title 
              :experiment-id experiment-id
              :run-info run-info
              :run-id run-id))
     (println "No mlflow URL provided. Logging disabled."))))

;(def n (prepare-experiment (make-network [1923 256 10]) "MXNet DeepBook Clojure implementation"))
;(log-param n "test" "del me")

;(.deleteExperiment (:mlflow n) "4")

(defn log-param
  ([net param-name param-val]
   (if (:run-id net)
     (.logParam (:mlflow net) (:run-id net) param-name (str param-val))
     (println "No Run ID provided."))))

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
   (let [sigmoid_z (sigmoid z)
         dyn-ones (ndarray/ones (ndarray/shape-vec z))]
   (ndarray/* sigmoid_z (ndarray/- dyn-ones sigmoid_z)))))




;def feedforward(self, a):
;        """Return the output of the network if "a" is input."""
;        for b, w in zip(self.biases, self.weights):
;            a = sigmoid(np.dot(w, a)+b)
;        return a


(defn feedforward ([^Network n
                    ^org.apache.mxnet.NDArray arr] ; image data
                   "Return the output of the network if a is input."
                   (let [args (map vector (:biases n) (:weights n))]
                       ;(println "Count Args: " (count args)) ; FIXME: WE HAVE EMPTY args !!!!!
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
   (let [;nabla_b (copy0 (:biases n))  ; seems we don't need this initialization!!!
         ;nabla_w (copy0 (:weights n))
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
                (cost_derivative (last activations) y) ; probably we can use peek instead of last?
                (sigmoid_prime (last zs)))
         nabla_b_last delta
         ; second from the tail, nabla_w_last is ok
         nabla_w_last (ndarray/dot delta (ndarray/transpose (nth activations (- (count activations) 2))))
         [nabla_b nabla_w] (apply mapv vector 
                                  (reductions (fn [[nb nw] [z w a]]
                                                (let [sp (sigmoid_prime z)
                                                      delta (ndarray/* (ndarray/dot (ndarray/transpose w) nb) sp)
                                                      ret_nw (ndarray/dot delta (ndarray/transpose a))]
                                                  [delta ret_nw]))
                                              [nabla_b_last nabla_w_last]
                                              (map vector 
                                                   (nthrest (reverse zs) 1) ; starts with -2
                                                   (reverse (nthrest (:weights n) 1)) ; starts with -1, skip rwal first
                                                   (nthrest (reverse activations) 2)))) ; starts with -3
         ]
    ; (clojure.pprint/pprint (take 10 (ndarray/->vec (first nabla_b))))
     ; (clojure.pprint/pprint (ndarray/to-scalar  (ndarray/max (first (reverse nabla_w)))))
     [(reverse nabla_b) (reverse nabla_w)])))

;(mapv #(vector %1 %2) ["Ford" "Arthur" "Tricia"] ["a" "b" "c"])
;(apply mapv vector [[1 "a"], [2 "b"], [3 "c"]])
;(mapv vector [1 "a"], [2 "b"], [3 "c"])


;(ndarray/->vec (ndarray/array [0.01 0.02 0.03 0.045 0.05 0.06][ 3 2]))
;(def a (ndarray/+ (ndarray/array [0.01 0.02 0.03 0.045 0.05 0.06][2 3]) 10))

;(def a (ndarray/* (ndarray/ones [1024 3000]) 0.3))

;(ndarray/->vec (ndarray/max a))
;
;(ndarray/->vec (ndarray/dot (ndarray/array [0.01 0.02 0.03 0.045 0.05 0.06][2 3]) 
;(ndarray/array [0.0003 0.0003 0.0003 0.0003 0.0003 0.0003] [2 3])))


(defn layers-det 
"For a list of ndarrays returns max() for each array"
([l]
(map (fn [a] (ndarray/to-scalar (ndarray/max a))) l)))

; NOTE: returns NEW network with updated weights and biases !!!
(defn update-mini-batch 
  ([^Network n mini_batch eta]
   (let [eta_by_minibatch_cnt (float (/ eta (count mini_batch)))
         calc (fn [w nw] (let [ m (ndarray/div nw (/ 1.0 eta_by_minibatch_cnt))
                                ;m (ndarray/* nw  eta_by_minibatch_cnt); this one will result in zero matrix !!!
                                ;_ (println "INTERMEDIATE"  (ndarray/to-scalar (ndarray/max nw)) "*" eta_by_minibatch_cnt "=" (ndarray/to-scalar (ndarray/max m)))
                                res (ndarray/- w m)]
                          res))
         nabla_b (copy0 (:biases n)) ; [ndarray nfarray ndarray]
         nabla_w (copy0 (:weights n)); [ndarray nfarray ndarray]
         [r_b r_w] (reduce (fn [[b w] [x y]] 
                             (let [[delta_nabla_b delta_nabla_w] (backprop n x y)] 
                               [(map ndarray/+ b delta_nabla_b) (map ndarray/+ w delta_nabla_w)]))
                           [nabla_b nabla_w]
                           mini_batch) ; [[ndarray ndarray] [ndarray ndarray]]
         new_net (assoc n 
                        :weights (mapv calc (:weights n) r_w) 
                        :biases (mapv calc (:biases n) r_b))
         ] 
       new_net))
  ([^Network n mini_batch eta debug-idx total]
(let [cnt (count mini_batch)]
   (do (if (= (rem (* cnt debug-idx) 1000) 0)
                                         (do (print (* debug-idx cnt))
                                        (print "/")
                                        (print total))
                                         (print "."))
       (flush)
       (update-mini-batch n mini_batch eta)))))


;(ndarray/->vec (ndarray/* (sigmoid (ndarray/ones [2 2])) 2.0))

(defn evaluate-network
  ([^Network n test_data]
   ; argmax(feedforward) returns ndarray with one element = index of prediction from 0 to 9
   (let [test_results (map (fn [[x y]] [(-> (ndarray/argmax (feedforward n x) 0)
                                            (ndarray/to-scalar)
                                            (int)) y]) 
                            test_data)]
     (count (filter (fn [[x y]]
                      (do
                        ;(clojure.pprint/pprint (ndarray/->vec x))
                        ;(println "-------------")
                        ;(clojure.pprint/pprint (ndarray/->vec y))
                        ;(println "=======================================")
                        (= x y))) test_results)))))


;   def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
(defn SGD 
  ([^Network net training_data epochs mini_batch_size eta] 
   (SGD net training_data epochs mini_batch_size eta nil))
  
  ([^Network net training_data epochs mini_batch_size eta test_data]
   (let [n_test (count test_data)
        training_cnt (count training_data)]
     (reduce (fn [n1 j]
       (let [mini_batches (partition-all mini_batch_size (shuffle training_data))
             _ (println "Mini batches: " (count mini_batches))
             [_ n] (reduce (fn [[idx n2] batch] [(+ 1 idx) (update-mini-batch n2 batch eta idx training_cnt)])
                                 [1 n1] (take 100 mini_batches))]
           (println)         
           (if (> n_test 0)
           (do 
              (printf "Epoch %d: %s / %d\n" j (evaluate-network n test_data) n_test )
              ;(clojure.pprint/pprint (take 10 (ndarray/->vec (first (:biases n)))))
           )
           (printf "Epoch %d complete\n" j))
           n
         ))
      net (range epochs)))))

; (get-experiment-id (make-network [784 30 10]) "DEL ME")


(defn run-net
  "First chapter of the book"
  [& [{epochs :epochs 
       mlflow-tracking-url :mlflow}]]
  (println "Training with code from Chapter 1")
  (println "Epochs: " epochs)

  (let [net (make-network [784 30 10] mlflow-tracking-url)
        net (prepare-experiment net "MXNet DeepBook Clojure implementation")
        all-data (load-data-wrapper)
        train-data (nth all-data 0)
        test-data (nth all-data 2)
        mini-batch-size 10
        eta 3.0
        ]
    (do
      (log-param net "Epochs" epochs)
      (log-param net "Mini Batch Size" mini-batch-size)
      (log-param net "Learning speed" eta)
      (SGD net train-data epochs mini-batch-size eta (take 1000 test-data))
      )))

