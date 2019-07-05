(ns deep-book.core
  (:require [deep-book.ch1])
  (:require [clojure.string :as string])
  (:require [clojure.tools.cli :refer [parse-opts]])
  (:gen-class))

(def cli-options
  [;; First three strings describe a short-option, long-option with optional
   ;; example argument description, and a description. All three are optional
   ;; and positional.
   ["-c" "--chapter NUM" "Chapter from the book to run code for"
    :default 1
    :parse-fn #(Integer/parseInt %)
    :validate [#(< 0 % 2) "Must be 1"]]
   ["-e" "--epochs NUM" "Number of Epochs"
    :default 30
    :parse-fn #(Integer/parseInt %)
    :validate [#(< 0 % 100) "Must be in between of 1 to 99"]]
   ["-v" nil "Verbosity level; may be specified multiple times to increase value"
    ;; If no long-option is specified, an option :id must be given
    :id :verbosity
    :default 0
    ;; Use :update-fn to create non-idempotent options (:default is applied first)
    :update-fn inc]
   ["-h" "--help"]])

(defn usage [options-summary]
  (->> ["This is Neural Network implementations from the free online book"
        "http://neuralnetworksanddeeplearning.com/"
        ""
        "Usage: lein run -- [options]"
        ""
        "Options:"
        options-summary]
       (string/join \newline)))

(defn error-msg [errors]
  (str "The following errors occurred while parsing your command:\n\n"
       (string/join \newline errors)))

(defn validate-args
  "Validate command line arguments. Either return a map indicating the program
  should exit (with a error message, and optional ok status), or a map
  indicating the action the program should take and the options provided."
  [args]
  (let [{:keys [options arguments errors summary]} (parse-opts args cli-options)]
    (cond
      (:help options) ; help => exit OK with usage summary
      {:exit-message (usage summary) :ok? true}
      errors ; errors => exit with description of errors
      {:exit-message (error-msg errors)}
      (:chapter options)
      {:chapter (:chapter options) :options options}
      :else ; failed custom validation => exit with usage summary
      {:exit-message (usage summary)})))

(defn exit [status msg]
  (println msg)
  (System/exit status))

(defn -main [& args]
  (let [{:keys [chapter options exit-message ok?]} (validate-args args)]
    (if exit-message
      (exit (if ok? 0 1) exit-message)
      (case chapter
        1  (deep-book.ch1/run-net options)))))