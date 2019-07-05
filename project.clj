(let [properties (select-keys (into {} (System/getProperties))
                              ["os.name"])
      platform (apply format "%s" (vals properties))

      ; https://stackoverflow.com/questions/4688336/what-is-an-elegant-way-to-set-up-a-leiningen-project-that-requires-different-dep
      mxnet (case platform
              "Mac OS X" '[org.apache.mxnet.contrib.clojure/clojure-mxnet-osx-cpu "1.4.1"]
              '[org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-cpu "1.4.1"])
      ; _ (println (str platform mxnet))
      ]

(defproject deep-book "0.1.0-SNAPSHOT"
  :description "MXNet Clojure version of the code for the 'Neural Networks and Deep Learning' free book"
  :url "https://github.com/deem0n/deep-book-clojure"
  :license {:name "MIT"
            :url "https://github.com/deem0n/deep-book-clojure/blob/master/LICENSE"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 ~mxnet]
  :main ^:skip-aot deep-book.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}}
  :jvm-opts ["-Xmx2g" "-server" "-XX:+UseConcMarkSweepGC" 
                                "-XX:+UseCompressedOops"
                                "-XX:+DoEscapeAnalysis"
                                "-XX:+UseBiasedLocking"] 
))


; java -jar deep-book-clojure-0.1.0-standalone.jar [args]
