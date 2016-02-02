# anomaly_detection.py
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
#from pyspark.ml.feature import StringIndexer,OneHotEncoder
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import operator

conf  = SparkConf().setAppName("anomaly detection")
sc    = SparkContext(conf=conf)
sqlCt = SQLContext(sc)

class AnomalyDetection():


    def readData(self, filename):
        self.rawDF = sqlCt.read.parquet(filename).cache()

    def readToyData(self):
        data = [(0, ["http", "udt", 0.4]),
                (1, ["http", "udf", 0.5]),
                (2, ["http", "tcp", 0.5]),
                (3, ["ftp", "icmp", 0.1]),
                (4, ["http", "tcp", 0.4])]
        schema = ["id", "rawFeatures"]
        self.rawDF = sqlCt.createDataFrame(data, schema)



    def cat2Num(self, df, indices):

        def encodeVectors(rows, str1, str2):
            new_rows = []
            count = len(rows)
            for row_number, row in enumerate(rows):
                hot_vector = [0]*count
                hot_vector[row_number] = 1
                new_rows.append((row[0], hot_vector))
            schema = [str1, str2]
            return sqlCt.createDataFrame(new_rows, schema)

        df = df.withColumn("features", df.rawFeatures)

        #Check indices in reverse order
        for ind in sorted(indices,reverse=True):
            # Extracting all categories
            df = df.withColumn(str(ind), df.rawFeatures.getItem(ind))
            df.cache()


            # Dataframe of categorical values
            cat_df = df.select(df[str(ind)]).distinct()

            # Was testing out some functions from the Spark Ml library.
            # They work, but it seems these functions aren't part of the cluster
            # API.
            '''
            stringIndexer = StringIndexer(inputCol=str(ind),outputCol=str(ind)+'_ind')
            encoder = OneHotEncoder(inputCol=str(ind)+"_ind",outputCol=str(ind)+"_hot")
            sparseToList = udf(lambda vect: vect.toArray().tolist(), ArrayType(StringType()))

            model = stringIndexer.fit(cat_df)
            temp_df   = model.transform(cat_df)
            temp_df2  = encoder.transform(temp_df)
            cat_df = temp_df2.select(temp_df2[str(ind)], sparseToList(temp_df2[str(ind)+'_hot']).alias(str(ind)+'_hot'))

            print "Look at this"
            cat_df.show()
            '''
            # Will return the df with a new column signifying the hot vectors for each
            cat_df = encodeVectors(cat_df.collect(), str(ind), str(ind) + "_hot")

            # Replacing category with one hot encoded vector
            rep = udf(lambda f, h: f[:ind]+h+f[ind+1:], ArrayType(StringType()))

            new_df = df.join(cat_df, df[str(ind)] == cat_df[str(ind)], 'inner')
            new_df = new_df.withColumn("newFeatures", rep(new_df["features"], new_df[str(ind)+"_hot"]))
            new_df = new_df.select(new_df["id"], new_df["rawFeatures"], new_df["newFeatures"].alias("features"))

            df.unpersist()
            df = new_df

        return df



    def addScore(self, df):

        c_df = df.groupBy(df["prediction"]).count()
        c_df.cache()
        c_df.show()
        max_df = c_df.select(c_df["count"]).groupBy().max()
        min_df = c_df.select(c_df["count"]).groupBy().min()
        c_df.unpersist()

        c_df = c_df.join(max_df).join(min_df)
        score_cal = udf(lambda x, minVal, maxVal: 1.0*(maxVal-x)/(maxVal-minVal), FloatType())

        c_df = c_df.withColumn("score", score_cal(c_df["count"], c_df["min(count)"], c_df["max(count)"]))
        c_df = c_df.select(c_df["prediction"], c_df["score"])

        c_df.cache()
        c_df.show()

        df = df.join(c_df, ["prediction"])
        c_df.unpersist()

        df.show()
        return df



    def detect(self, k, t):
        #Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show()

        #Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)

        #Adding the prediction column to df1
        modelBC = sc.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show()

        #Adding the score column to df2; The higher the score, the more likely it is an anomaly
        df3 = self.addScore(df2).cache()
        df3.show()

        return df3.where(df3.score > t)




if __name__ == "__main__":
    ad = AnomalyDetection()
    #ad.readToyData()
    ad.readData("logs-features")
    anomalies = ad.detect(8, 0.97)
    print anomalies.count()
    anomalies.show()