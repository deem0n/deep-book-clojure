# deep-book-clojure

[MXNet Clojure](https://mxnet.incubator.apache.org/api/clojure/index.html) version of the code for the [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) free book 

## Installation

Note, that you will need to download and unzip MNIST images into the `data` directory. Use script from the MXNet project which I put into [utils/get_mnist_data.sh](utils/get_mnist_data.sh), it should do the right thing. The data files are about 50Mb in size, so I did'nt commit them to github.

```
git clone https://github.com/deem0n/deep-book-clojure.git
cd deep-book-clojure
utils/get_mnist_data.sh
```

## Usage

```
lein test
lein run
```

## Code comparasion

<table>
    <tr>
        <th>Python</th>
        <th>Clojure</th>
    </tr>
<tr>
<td>

  ```python
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
  ```
</td>
<td>

  ```clojure
(defrecord Network [^java.lang.Long num_layers
                    ^clojure.lang.PersistentVector sizes
                    ^clojure.lang.LazySeq biases
                    ^clojure.lang.LazySeq weights])

;Constructor
(defn make-network ([sizes]
                    (->Network
                     (count sizes)
                     sizes
                     (map #(random/normal 0 1 [% 1]) (subvec sizes 1))
                     (map #(random/normal 0 1 [%2 %1]) (butlast sizes) (subvec sizes 1)))))

  ```
</td>
</tr>
</table>

## License

Copyright © 2019 Dmitry Dorofeev

[MIT LICENSE](./LICENSE)
