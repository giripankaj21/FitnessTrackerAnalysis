package firstspark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import java.io.File



object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Fitness Tracker Analysis")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val filePath = "dataset/project18_fitness_data.csv"
    val fitnessDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(filePath)

    println("=== Sample Data ===")
    fitnessDF.show(5)

    println("=== Summary Statistics ===")
    fitnessDF.describe().show()

    println("=== Average by DayType ===")
    val avgDayType = fitnessDF.groupBy("DayType")
      .agg(
        avg("Steps").alias("AvgSteps"),
        avg("CaloriesBurned").alias("AvgCalories"),
        avg("ActiveMinutes").alias("AvgActiveMinutes")
      )

    avgDayType.show()

    println("=== Top 5 Users by Steps ===")
    fitnessDF.orderBy(desc("Steps")).show(5)

    println("=== Top 5 Users by Calories Burned ===")
    fitnessDF.orderBy(desc("CaloriesBurned")).show(5)

    println("=== Top 5 Users by Active Minutes ===")
    fitnessDF.orderBy(desc("ActiveMinutes")).show(5)

    println("=== Weekday vs Weekend Comparison ===")
    fitnessDF.groupBy("DayType")
      .agg(
        round(avg("Steps"), 2).alias("AvgSteps"),
        round(avg("CaloriesBurned"), 2).alias("AvgCaloriesBurned"),
        round(avg("ActiveMinutes"), 2).alias("AvgActiveMinutes")
      ).show()

    val scoredDF = fitnessDF.withColumn(
      "ActivityScore",
      round(
        col("Steps") * 0.001 +
          col("CaloriesBurned") * 0.1 +
          col("ActiveMinutes") * 0.2, 2
      )
    )

    println("=== Top 5 Most Active Users (Activity Score) ===")
    scoredDF.orderBy(desc("ActivityScore")).show(5)

    // Select relevant columns
    val featureCols = Array("Steps", "CaloriesBurned", "ActiveMinutes")
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val assembledDF = assembler.transform(fitnessDF)

    // KMeans clustering
    val kmeans = new KMeans().setK(3).setSeed(1L)
    val model = kmeans.fit(assembledDF)

    val predictions = model.transform(assembledDF)

    println("=== Cluster Centers ===")
    model.clusterCenters.foreach(center => println(center))

    println("=== Users with Cluster Labels ===")
    predictions.select("UserID", "Steps", "CaloriesBurned", "ActiveMinutes", "prediction").show(10)

    // Index DayType (label column)
    val labelIndexer = new StringIndexer()
      .setInputCol("DayType")
      .setOutputCol("label")
      .fit(fitnessDF)

    val labeledDF = labelIndexer.transform(assembledDF)

    // Split into training and test data
    val Array(trainingData, testData) = labeledDF.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Define Decision Tree Classifier
    val dt = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")

    // Train the model
    val dtModel = dt.fit(trainingData)

    // Make predictions
    val dtpredictions = dtModel.transform(testData)

    // Evaluate the model
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(dtpredictions)
    println(s"=== Decision Tree Model Accuracy: ${accuracy * 100} %")

    // Show sample predictions
    dtpredictions.select("UserID", "Steps", "CaloriesBurned", "ActiveMinutes", "DayType", "prediction").show(10)


    val summary=testData.groupBy("DayType")
      .agg(
        avg("Steps").alias("AvgSteps"),
        avg("CaloriesBurned").alias("AvgCaloriesBurned"),
        avg("ActiveMinutes").alias("AvgActiveMinutes"),
        count("*").alias("TotalRecords")
      )

    println("=== Fitness Summary by Day Type ===")
    summary.show()

    // Create output directory if not exists (local filesystem only)
    new File("output").mkdirs()

    // 1. Export KMeans Clustered Data
    val clusteredOutput = predictions.select("UserID", "Steps", "CaloriesBurned", "ActiveMinutes", "prediction")
    clusteredOutput.coalesce(1) // optional: reduce to 1 file
      .write.option("header", "true").mode("overwrite")
      .csv("output/kmeans_clustered_data")

    // 2. Export Decision Tree Predictions
    val dtOutput = dtpredictions.select("UserID", "Steps", "CaloriesBurned", "ActiveMinutes", "DayType", "prediction")
    dtOutput.coalesce(1)
      .write.option("header", "true").mode("overwrite")
      .csv("output/decision_tree_predictions")

    // 3. Export Activity Scores
    val activityOutput = scoredDF.select("UserID", "Steps", "CaloriesBurned", "ActiveMinutes", "ActivityScore")
    activityOutput.coalesce(1)
      .write.option("header", "true").mode("overwrite")
      .csv("output/activity_scores")

    println("Export completed. Check the 'output' folder for CSVs.")

    spark.stop()
  }
}
