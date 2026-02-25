from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def main():
    spark = (
        SparkSession.builder
        .appName("Titanic-LogReg-MLlib")
        .enableHiveSupport()
        .getOrCreate()
    )

    # 1) Read from Hive (Objective 5 requirement: use your Hive table)
    df = spark.sql("SELECT * FROM finalproject.titanic_raw")

    # 2) Basic cleaning / type casting (table stored as STRINGs)
    df = (
        df
        .withColumn("Survived", col("Survived").cast("int"))
        .withColumn("Pclass", col("Pclass").cast("int"))
        .withColumn("Age", col("Age").cast("double"))
        .withColumn("SibSp", col("SibSp").cast("int"))
        .withColumn("Parch", col("Parch").cast("int"))
        .withColumn("Fare", col("Fare").cast("double"))
        .withColumn("Sex", trim(col("Sex")))
        .withColumn("Embarked", trim(col("Embarked")))
    )

    # Handle missing values
    # - Age, Fare: fill with 0 (simple + safe for this assignment)
    # - Embarked: fill unknown category
    df = df.fillna({"Age": 0.0, "Fare": 0.0, "Embarked": "UNK", "Sex": "UNK"})

    # Some rows may still have null label (Survived) -> drop them
    df = df.filter(col("Survived").isNotNull())

    # 3) Train/test split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # 4) Feature engineering pipeline
    sex_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_idx", handleInvalid="keep")
    embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="Embarked_idx", handleInvalid="keep")

    encoder = OneHotEncoder(
        inputCols=["Sex_idx", "Embarked_idx"],
        outputCols=["Sex_ohe", "Embarked_ohe"]
    )

    assembler = VectorAssembler(
        inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_ohe", "Embarked_ohe"],
        outputCol="features"
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="Survived",
        predictionCol="prediction",
        probabilityCol="probability",
        maxIter=50,
        regParam=0.0
    )

    pipeline = Pipeline(stages=[sex_indexer, embarked_indexer, encoder, assembler, lr])

    # 5) Fit model
    print("=== Training Logistic Regression model ===")
    model = pipeline.fit(train_df)

    # 6) Predict
    predictions = model.transform(test_df)

    # Show a few predictions for “training output” evidence
    print("=== Sample Predictions (label vs prediction) ===")
    predictions.select("Survived", "prediction", "probability", "Pclass", "Sex", "Age", "Fare", "Embarked").show(20, truncate=False)

    # 7) Metrics (Objective 5 requirement)
    # AUC (works even if classes are imbalanced)
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="Survived",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = auc_evaluator.evaluate(predictions)

    # Accuracy
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="Survived",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = acc_evaluator.evaluate(predictions)

    print("=== Evaluation Metrics ===")
    print(f"AUC (areaUnderROC): {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()