from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import happybase

# ---- Write metrics to HBase with happybase 
def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('titanic_metrics')  
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

def main():
    # Step 1: Create Spark session 
    spark = SparkSession.builder.appName("Titanic Survival Prediction").enableHiveSupport().getOrCreate()

    # Step 2: Load data from Hive table 
    titanic_df = spark.sql("""
        SELECT
          cast(Pclass as int)       as Pclass,
          trim(Sex)                as Sex,
          cast(Age as double)      as Age,
          cast(SibSp as int)       as SibSp,
          cast(Parch as int)       as Parch,
          cast(Fare as double)     as Fare,
          trim(Embarked)           as Embarked,
          cast(Survived as int)    as Survived
        FROM finalproject.titanic_raw
    """)

    # Step 3: Handle null values
    titanic_df = titanic_df.filter(col("Survived").isNotNull())
    titanic_df = titanic_df.fillna({"Age": 0.0, "Fare": 0.0, "Sex": "UNK", "Embarked": "UNK"})

    # Step 4: Feature prep (categorical -> index -> onehot, then assemble)
    sex_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_idx", handleInvalid="keep")
    emb_indexer = StringIndexer(inputCol="Embarked", outputCol="Embarked_idx", handleInvalid="keep")

    encoder = OneHotEncoder(
        inputCols=["Sex_idx", "Embarked_idx"],
        outputCols=["Sex_ohe", "Embarked_ohe"]
    )

    assembler = VectorAssembler(
        inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_ohe", "Embarked_ohe"],
        outputCol="features",
        handleInvalid="skip"
    )

    # Step 5: Split train/test
    train_df, test_df = titanic_df.randomSplit([0.7, 0.3], seed=42)

    # Step 6: Train MLlib model (Logistic Regression is relevant for Survived 0/1)
    lr = LogisticRegression(featuresCol="features", labelCol="Survived", maxIter=50)

    pipeline = Pipeline(stages=[sex_indexer, emb_indexer, encoder, assembler, lr])

    print("=== Training Logistic Regression model ===")
    model = pipeline.fit(train_df)

    # Step 7: Predict
    predictions = model.transform(test_df)

    print("=== Sample Predictions ===")
    predictions.select("Survived", "prediction", "probability", "Pclass", "Sex", "Age", "Fare", "Embarked") \
               .show(20, truncate=False)

    # Step 8: Evaluation metrics (classification metrics)
    auc_eval = BinaryClassificationEvaluator(
        labelCol="Survived",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = auc_eval.evaluate(predictions)

    acc_eval = MulticlassClassificationEvaluator(
        labelCol="Survived",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = acc_eval.evaluate(predictions)

    print("=== Evaluation Metrics ===")
    print(f"AUC: {auc}")
    print(f"Accuracy: {accuracy}")

    # Write metrics to HBase 
    data = [
        ("metrics1", "cf:auc", str(auc)),
        ("metrics1", "cf:accuracy", str(accuracy)),
    ]

    rdd = spark.sparkContext.parallelize(data)
    rdd.foreachPartition(write_to_hbase_partition)

    # Step 9: Stop spark
    spark.stop()

if __name__ == "__main__":
    main()
