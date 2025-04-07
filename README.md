
# Fitness Tracker Activity Analysis

This Apache Spark-based Scala project analyzes fitness activity data using IntelliJ IDEA without SBT. It includes data processing, clustering with KMeans, classification with Decision Tree, and summary reporting.

## 📁 Project Structure

```
FitnessTrackerAnalysis/
│
├── data/
│   └── fitness_data.csv              # Input dataset
│
├── output/
│   ├── summary.csv                   # Summary statistics
│   ├── top_users_by_steps.csv        # Top users by steps
│   ├── top_users_by_calories.csv     # Top users by calories
│   ├── top_users_by_minutes.csv      # Top users by active minutes
│   ├── cluster_centers.csv           # Cluster centers
│   ├── clustered_users.csv           # Cluster predictions
│   ├── decision_tree_predictions.csv # Decision tree results
│   └── fitness_summary_by_day.csv    # Final summary by day type
│
├── lib/
│   └── spark-core, spark-sql JARs    # Manually added Spark dependencies
│
└── FitnessAnalysis.scala             # Main Scala source code
```

## 🔧 Technologies Used
- Apache Spark (Core & SQL)
- Scala (in IntelliJ IDEA)
- CSV Data Processing
- KMeans Clustering
- Decision Tree Classification

## 🚀 How to Run

1. Open `IntelliJ IDEA` and create a new Scala project (no SBT).
2. Add Spark JARs manually to the project `lib` folder and set them in module dependencies.
3. Place `fitness_data.csv` in `data/`.
4. Copy code from `FitnessAnalysis.scala` and run.

## 📊 Output Summary

- **Basic Statistics**
  - Mean, StdDev, Min/Max of Steps, Calories, Minutes
- **DayType Comparison**
  - Weekday vs Weekend averages
- **Top Users**
  - Based on Steps, Calories, Minutes
- **KMeans Clustering**
  - Groups users into 3 clusters
- **Decision Tree**
  - Predicts DayType from fitness data

## 📎 Sample Output

```
=== Summary Statistics ===
Mean Steps: 11177.52 | Calories: 437.10 | Minutes: 92.41

=== Day Type Comparison ===
Weekday - Steps: 11219.98, Calories: 424.28, Minutes: 93.05
Weekend - Steps: 11146.77, Calories: 446.38, Minutes: 91.95

=== Cluster Centers ===
[10304.60, 447.87, 88.00]
[16608.78, 440.05, 93.29]
[4388.49, 422.91, 95.31]

=== Decision Tree Accuracy ===
59.45%
```

## 📁 Author

Developed by [Giri Pankaj](https://github.com/giripankaj21) as part of a hands-on data analysis project for INT315.

---

This project provides a solid foundation for scalable fitness data analysis using Spark and Scala. Ideal for learning clustering, classification, and data summarization.
