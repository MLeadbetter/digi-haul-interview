import polars as pl
import numpy as np
from catboost import CatBoostClassifier, cv, Pool
from numbers import Number
from typing import Optional


IS_CLOSE_PROXIMITY_KM = 1
MIN_GPS_POINTS = 50
EARTH_RADIUS_KM = 6371
SPEED_LIMIT_KMPH = 100
DATE_TIME_COLUMNS = ["FIRST_COLLECTION_SCHEDULE_LATEST", "LAST_DELIVERY_SCHEDULE_LATEST",
                     "FIRST_COLLECTION_SCHEDULE_EARLIEST", "LAST_DELIVERY_SCHEDULE_EARLIEST"]


def haversine(theta: Number) -> float:
    return np.sin(theta / 2)**2


def polar_distance(lat1: pl.Series, lon1: pl.Series, lat2: pl.Series, lon2: pl.Series) -> pl.Series:
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return 2*EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine(dlat) 
                                         + (1 - haversine(dlat) - haversine(lat2 + lat1)) * haversine(dlon)))


def nano_to_hours(nanos: Number) -> float:
    return nanos / (10**9 * 60 * 60)


def clean_gps(gps: pl.DataFrame) -> pl.DataFrame:
    gps = gps.sort(["SHIPMENT_NUMBER", "RECORD_TIMESTAMP"]).with_columns(pl.col("RECORD_TIMESTAMP").shift().over("SHIPMENT_NUMBER").alias("prev_TIMESTAMP"))
    gps = gps.with_columns((pl.col("RECORD_TIMESTAMP") - pl.col("prev_TIMESTAMP")).alias("time_diff"))
    gps = gps.with_columns(pl.col("LAT").shift().over("SHIPMENT_NUMBER").alias("prev_LAT"))
    gps = gps.with_columns(pl.col("LON").shift().over("SHIPMENT_NUMBER").alias("prev_LON"))
    
    gps_distances = polar_distance(gps.select("LAT"), gps.select("LON"), gps.select("prev_LAT"), gps.select("prev_LON"))
    gps = gps.with_columns(pl.Series(name="distance_since_prev", values=gps_distances.ravel()))
    gps = gps.with_columns((pl.col("distance_since_prev") / nano_to_hours(pl.col("time_diff").dt.total_nanoseconds())).alias("speed_kmph"))
    
    gps_filtered = gps.filter((pl.col("speed_kmph") < SPEED_LIMIT_KMPH) & (pl.col("speed_kmph") > 0))
    
    used_shipment_numbers = gps_filtered.group_by("SHIPMENT_NUMBER").agg(pl.col("LAT").count().alias("gps_points_after"))\
                                       .filter(pl.col("gps_points_after") > MIN_GPS_POINTS).select("SHIPMENT_NUMBER")
    
    return used_shipment_numbers.join(gps_filtered, on="SHIPMENT_NUMBER").select(["SHIPMENT_NUMBER", "LAT", "LON", "RECORD_TIMESTAMP", "time_diff", "speed_kmph"])


def get_arrival_times(bookings: pl.DataFrame, gps: pl.DataFrame) -> pl.DataFrame:
    distance_data = gps.join(bookings, on="SHIPMENT_NUMBER").select(["SHIPMENT_NUMBER", "LAT", "LON", "RECORD_TIMESTAMP", "LAST_DELIVERY_LATITUDE", 
                                                                     "LAST_DELIVERY_LONGITUDE", "LAST_DELIVERY_SCHEDULE_LATEST"])
    distances = polar_distance(distance_data.select("LAT"), distance_data.select("LON"), distance_data.select("LAST_DELIVERY_LATITUDE"),
                               distance_data.select("LAST_DELIVERY_LONGITUDE"))
    distance_data = distance_data.with_columns(pl.Series(name="distance_to_delivery", values=distances.ravel()))
    close_points = distance_data.filter(pl.col("distance_to_delivery") < IS_CLOSE_PROXIMITY_KM)
    arrivals = close_points.group_by("SHIPMENT_NUMBER").agg(pl.col("RECORD_TIMESTAMP").min().alias("arrival"))\
                           .join(bookings, on="SHIPMENT_NUMBER").select(["SHIPMENT_NUMBER", "arrival", "LAST_DELIVERY_SCHEDULE_LATEST"])
    return arrivals.with_columns((pl.col("LAST_DELIVERY_SCHEDULE_LATEST") < pl.col("arrival")).alias("is_late"))

def date_time_features(df: pl.DataFrame, field: str, use_seconds_since_midnight: bool = True, inter_day_features: bool = False) -> pl.DataFrame:
    dt_series = df.select(field).to_series()
    feature_name = field + "_"

    new_features = []
    if use_seconds_since_midnight:
        seconds = dt_series.dt.second().cast(pl.UInt32)
        minutes = dt_series.dt.minute().cast(pl.UInt32)
        hour = dt_series.dt.hour().cast(pl.UInt32)
        seconds_since_midnight = (seconds + 60*(minutes + 60*hour)).alias(feature_name + "seconds_since_midnight")
        new_features.append(seconds_since_midnight)
    if inter_day_features:
        weekday = dt_series.dt.weekday().alias(feature_name + "weekday")
        new_features.append(weekday)

        date = dt_series.dt.day().alias(feature_name + "date")
        new_features.append(date)

        month = dt_series.dt.month().alias(feature_name + "month")
        new_features.append(month)

        #year = dt_series.dt.year().alias(feature_name + "year")
        #new_features.append(year)
    
    return pl.DataFrame(new_features)


def add_booking_features(bookings: pl.DataFrame) -> pl.DataFrame:
    bookings = bookings.with_columns(pl.col("FIRST_COLLECTION_POST_CODE").str.split(' ').list.get(0).alias("collection_post_code_sector"))
    bookings = bookings.with_columns(pl.col("LAST_DELIVERY_POST_CODE").str.split(' ').list.get(0).alias("delivery_post_code_sector"))
    bookings = bookings.with_columns(pl.col("FIRST_COLLECTION_POST_CODE").fill_null(strategy="zero"))
    bookings = bookings.with_columns(pl.col("LAST_DELIVERY_POST_CODE").fill_null(strategy="zero"))
    bookings = bookings.with_columns(pl.col("collection_post_code_sector").fill_null(strategy="zero"))
    bookings = bookings.with_columns(pl.col("delivery_post_code_sector").fill_null(strategy="zero"))
    crow_distance = polar_distance(bookings.select(pl.col("FIRST_COLLECTION_LATITUDE")), bookings.select(pl.col("FIRST_COLLECTION_LONGITUDE")),
                                   bookings.select(pl.col("LAST_DELIVERY_LATITUDE")), bookings.select(pl.col("LAST_DELIVERY_LONGITUDE")))
    bookings = bookings.with_columns((pl.col("LAST_DELIVERY_SCHEDULE_LATEST") - pl.col("FIRST_COLLECTION_SCHEDULE_EARLIEST"))
                                     .dt.total_seconds().alias("available_time_early_late"))
    bookings = bookings.with_columns((pl.col("LAST_DELIVERY_SCHEDULE_LATEST") - pl.col("FIRST_COLLECTION_SCHEDULE_LATEST"))
                                     .dt.total_seconds().alias("available_time_late_late"))
    bookings = bookings.with_columns(crow_distance = np.ravel(crow_distance))
    bookings_with_times = bookings
    for date_time_column in DATE_TIME_COLUMNS[:-1]:
        bookings_with_times = bookings_with_times.with_columns(date_time_features(bookings, date_time_column))
    
    return bookings_with_times.with_columns(date_time_features(bookings, "LAST_DELIVERY_SCHEDULE_EARLIEST", inter_day_features=False))


def create_features(bookings: pl.DataFrame, arrivals: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    bookings = add_booking_features(bookings)
    if arrivals is not None:
        bookings = arrivals.select("SHIPMENT_NUMBER", "is_late").join(bookings, on="SHIPMENT_NUMBER")
        #bookings = bookings.select(pl.exclude(["is_late"]))
    X = bookings.select(pl.exclude(["PROJECT_ID", "SHIPMENT_NUMBER", "is_late"] + DATE_TIME_COLUMNS))

    if arrivals is not None:
        y = bookings.select("is_late")
        return X, y
    return X


def get_cat_features(X: pl.DataFrame) -> list:
    cat_features = []
    for i, dtype in enumerate(X.dtypes):
        if dtype == pl.String:
            cat_features.append(i)
    return cat_features


def train_predict(X: pl.DataFrame, y: pl.DataFrame, X_test: pl.DataFrame) -> pl.Series:
    # parameters for training inside cv:
    train_pool = Pool(data=X.to_pandas(), label=y.to_pandas(), cat_features=get_cat_features(X), has_header=True)
    
    cbc = CatBoostClassifier(iterations=2000, learning_rate=0.01)
    cbc.fit(train_pool, verbose=True)
    return pl.Series(cbc.predict_proba(X_test.to_pandas())[:,0]).alias("likelihood_late")

if __name__ == "__main__":
    bookings = pl.read_csv("Shipment_bookings.csv", try_parse_dates=True)
    gps = pl.read_csv("GPS_data.csv", try_parse_dates=True)
    to_predict = pl.read_csv("New_bookings.csv", try_parse_dates=True).select(pl.exclude("SHIPPER_ID"))
    to_predict = to_predict.rename({"CARRIER_ID": "CARRIER_DISPLAY_ID"})

    arrivals = get_arrival_times(bookings, gps)
    X, y = create_features(bookings, arrivals)
    X_test = create_features(to_predict)
    predictions = train_predict(X, y, X_test)
    id_preds = pl.DataFrame([to_predict.select(pl.col("SHIPMENT_NUMBER")).to_series(), predictions])
    id_preds.write_csv("predictions.csv")
