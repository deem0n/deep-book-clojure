(ns deep-book.mlflow)

(defn get-experiment-id
  [^org.mlflow.tracking.MlflowClient mlflow experimaent-name]
  (let [optional-exp-id (.getExperimentByName mlflow experimaent-name)]
    (if (.isPresent optional-exp-id)
      (.. optional-exp-id (get) (getExperimentId))
      (.createExperiment mlflow experimaent-name))))