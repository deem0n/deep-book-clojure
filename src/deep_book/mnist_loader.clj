(ns deep-book.mnist-loader
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray])
  (:require [org.apache.clojure-mxnet.io :as mx-io]))

(def data-dir "data/")
; 60000 images in total
(defn train-data 
  ([batch-size]
   (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                      :label (str data-dir "train-labels-idx1-ubyte")
                      :label-name "softmax_label"
                      :input-shape [784]
                      :batch-size batch-size
                      :shuffle true
                      :flat true
                      :silent false
                      :seed 10
                      })))

; 10000 images in total
(defn test-data 
  ([batch-size]
   (mx-io/mnist-iter {:image (str data-dir "t10k-images-idx3-ubyte")
                      :label (str data-dir "t10k-labels-idx1-ubyte")
                      :input-shape [784]
                      :batch-size batch-size
                      :flat true
                      :silent false})))

;(mx-io/provide-data-desc train-data)


;(map #(ndarray/shape-vec (first (mx-io/batch-data %))) (mx-io/batches (train-data 10000)))

;(ndarray/shape-vec (first (mx-io/batch-label (mx-io/next train-data))) )
;clojure.lang.PersistentVector

(comment
(let [data (ndarray/array [2 1 0 -1 -2   3 4 5 6 7 ] [2 5])
      s (ndarray/slice-axis data 1 0 1)
      shape (ndarray/shape-vec s)
      ]
  (println (ndarray/->vec s))
  (println shape)
  (map #(ndarray/->vec (first (mx-io/batch-data %))) (mx-io/batches (mx-io/ndarray-iter [data])))
  )
)
   "Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."

(defn vectorized-result 
  ([j]
   (ndarray/array (assoc (vec (replicate 10 0)) j 1) [10 1])
   ))

(defn batch-to-tuples
  "converts batch to seq of tuples [a,b]. 
where a is ndarray with data, b is vectorized value from corresponding label.
Data is NDArray and has shape [10000 784], so we should iterate correctly over it"
  ([batch]
   (let [data-nda (first (mx-io/batch-data batch))
         labels-nda (first (mx-io/batch-label batch))
         data (map #(first (mx-io/batch-data %)) (mx-io/batches (mx-io/ndarray-iter [data-nda])))]
     (map (fn [d l] [(ndarray/transpose d) (vectorized-result (int l))]) data (ndarray/->vec labels-nda)))))

(defn batch-to-tuples-values
  "converts batch to seq of tuples [a,b]. 
where a is ndarray with data, b is value from corresponding label.
Data is NDArray and has shape [10000 784], so we should iterate correctly over it"
  ([batch]
   (let [data-nda (first (mx-io/batch-data batch))
         labels-nda (first (mx-io/batch-label batch))
         data (map #(first (mx-io/batch-data %)) (mx-io/batches (mx-io/ndarray-iter [data-nda])))]
     (map (fn [d l] [(ndarray/transpose d) (int l)]) data (ndarray/->vec labels-nda)))))



    "Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."

(defn load-data-wrapper
  ([]
   (let [train-batches (mx-io/batches (train-data 10000))
         test-batches (mx-io/batches (test-data 5000))
         train-and-val (partition-all 5 (map batch-to-tuples train-batches))
         ]

       [ (into [] (apply concat (first train-and-val))) ; train 50,000
        (first (second train-and-val)) ; validation 10,000
        (into [] (apply concat (map batch-to-tuples-values test-batches))) ; test 10,000
        ]

     )))


; (instance? clojure.lang.PersistentVector  (load-data-wrapper))
 ; (take 1 (take 1 (second (load-data-wrapper))))

 ; (take 1 (nth (load-data-wrapper) 0))
 ; (count (nth (load-data-wrapper) 0))
; (conj (into [] (partition-all 2  [[1 2 3 4] [5 6 7 8] [9 10 11]])) [100 111])

; (first  (partition-all 2  [[1 2 3 4] [5 6 7 8] [9 10 11]]))