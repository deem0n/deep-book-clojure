(let [properties (select-keys (into {} (System/getProperties))
                              ["os.name"])
      platform (apply format "%s" (vals properties))

      ; https://stackoverflow.com/questions/4688336/what-is-an-elegant-way-to-set-up-a-leiningen-project-that-requires-different-dep
      mxnet (case platform
              "Mac OS X" '[org.apache.mxnet.contrib.clojure/clojure-mxnet-osx-cpu "1.4.1"]
              '[org.apache.mxnet.contrib.clojure/clojure-mxnet "1.4.1-SNAPSHOT"]) ; this one is from local .m2 repository
              ;'[org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-cpu "1.4.1"]) ; this one form the remote repo
      ; _ (println (str platform mxnet))
      ]

(defproject deep-book "0.1.0-SNAPSHOT"
  :description "MXNet Clojure version of the code for the 'Neural Networks and Deep Learning' free book"
  :url "https://github.com/deem0n/deep-book-clojure"
  :license {:name "MIT"
            :url "https://github.com/deem0n/deep-book-clojure/blob/master/LICENSE"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [org.clojure/tools.cli "0.4.2"]
                 [org.mlflow/mlflow-client "1.0.0"]
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
