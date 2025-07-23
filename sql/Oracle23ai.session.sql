SELECT 
        *
FROM
    zuber_cabs



SELECT 
        *
FROM
    zuber_neighborhoods



SELECT 
        *
FROM
    ZUBER_TRIPS



SELECT 
        *
FROM
    zuber_weather_records



-- 1. Print the company_name field. 
--    Find the number of taxi trips for each taxi company for November 15 and 16, 2017, 
--    name the resulting field trips_amount, and print it as well. 
--    Sort the results by the trips_amount field in descending order.

SELECT zuber_cabs."company_name",
    COUNT(zuber_trips."trip_id") AS trips_amount
FROM 
    zuber_cabs
    INNER JOIN zuber_trips ON zuber_trips."cab_id" = zuber_cabs."cab_id"
WHERE 
    TRUNC(TO_DATE(zuber_trips."start_ts", 'YYYY-MM-DD HH24:MI:SS')) BETWEEN TO_DATE('2017-11-15', 'YYYY-MM-DD') 
                                                                    AND TO_DATE('2017-11-16', 'YYYY-MM-DD')
GROUP BY 
    zuber_cabs."company_name"
ORDER BY 
    trips_amount DESC;



-- 2. Find the number of trips for each taxi company whose name contains the words "Yellow" or "Blue" 
--    from November 1 to 7, 2017. 
--    Name the resulting variable trips_amount. 
--    Group the results by the company_name field.

SELECT zuber_cabs."company_name" AS company_name,
    COUNT(zuber_trips."trip_id") AS trips_amount
FROM 
    zuber_cabs
    INNER JOIN zuber_trips ON zuber_trips."cab_id" = zuber_cabs."cab_id"
WHERE 
    TRUNC(TO_DATE(zuber_trips."start_ts", 'YYYY-MM-DD HH24:MI:SS')) BETWEEN TO_DATE('2017-11-01', 'YYYY-MM-DD') 
                                                                    AND TO_DATE('2017-11-07', 'YYYY-MM-DD')
    AND zuber_cabs."company_name" LIKE '%Yellow%'
GROUP BY 
    zuber_cabs."company_name"

UNION ALL

SELECT zuber_cabs."company_name" AS company_name,
    COUNT(zuber_trips."trip_id") AS trips_amount
FROM 
    zuber_cabs
    INNER JOIN zuber_trips ON zuber_trips."cab_id" = zuber_cabs."cab_id"
WHERE 
    TRUNC(TO_DATE(zuber_trips."start_ts", 'YYYY-MM-DD HH24:MI:SS')) BETWEEN TO_DATE('2017-11-01', 'YYYY-MM-DD') 
                                                                    AND TO_DATE('2017-11-07', 'YYYY-MM-DD')
    AND zuber_cabs."company_name" LIKE '%Blue%'
GROUP BY 
    zuber_cabs."company_name";



-- 3. From November 1 to 7, 2017, the most popular taxi companies were Star North Management LLC and Taxi Affiliation Services. 
--   Find the number of trips from these two companies and name the resulting variable trips_amount. 
--   Group the trips from all other companies into the "Other" group. 
--   Group the data by taxi company names. 
--   Name the field with taxi company names "company." 
--   Sort the result in descending order by trips_amount.

SELECT
    CASE 
        WHEN zuber_cabs."company_name" = 'Star North Management LLC' THEN 'Star North Management LLC' 
        WHEN zuber_cabs."company_name" = 'Taxi Affiliation Services Yellow' THEN 'Taxi Affiliation Services Yellow'
        ELSE 'OTHER'
    END AS company,
    COUNT(zuber_trips."trip_id") AS trips_amount                
FROM 
    zuber_cabs
INNER JOIN zuber_trips ON zuber_trips."cab_id" = zuber_cabs."cab_id"
WHERE 
    TRUNC(TO_DATE(zuber_trips."start_ts", 'YYYY-MM-DD HH24:MI:SS')) BETWEEN TO_DATE('2017-11-01', 'YYYY-MM-DD') 
                                                                    AND TO_DATE('2017-11-07', 'YYYY-MM-DD')
GROUP BY 
    CASE 
        WHEN zuber_cabs."company_name" = 'Star North Management LLC' THEN 'Star North Management LLC' 
        WHEN zuber_cabs."company_name" = 'Taxi Affiliation Services Yellow' THEN 'Taxi Affiliation Services Yellow'
        ELSE 'OTHER'
    END
ORDER BY 
    trips_amount DESC;



-- 4. Retrieve the O'Hare and Loop neighborhood identifiers from the neighborhoods table.

SELECT "neighborhood_id",
    "name"
FROM 
    zuber_neighborhoods
WHERE 
    "name" LIKE '%Hare' OR "name" LIKE 'Loop'



-- 5. For each hour, retrieve the weather condition records from the weather_records table. 
--    Using the CASE operator, divide all hours into two groups: Bad if the description field contains the words rain or storm, and
--    Good for all others. 
--    Name the resulting field weather_conditions. 
--    The final table should include two fields: datetime (ts) and weather_conditions.

SELECT "Date and time",
    CASE
        WHEN DESCRIPTION LIKE '%rain%' OR DESCRIPTION LIKE '%storm%' THEN 'Bad'
        ELSE 'Good'
    END AS weather_conditions
FROM 
    zuber_weather_records;



-- 6. Retrieve from the trips table all trips that started at the Norwood Park (pickup_location_id: 62) on Saturday and ended at Rush & Division (dropoff_location_id: 72). 
--    Obtain the weather conditions for each trip. 
--    Use the method you used in the previous task. 
--    Also retrieve the duration of each trip. 
--    Ignore trips for which no weather data is available.
--    The table columns should be in the following order:
--        start_ts
--        weather_conditions
--        duration_seconds
--        Sort by trip_id.

SELECT 
    zt."trip_id",
    zt."start_ts",
    w.weather_conditions,
    zt."duration_seconds"
FROM 
    zuber_trips zt
INNER JOIN (
    SELECT 
        "Date and time" AS weather_ts,
        CASE
            WHEN DESCRIPTION LIKE '%rain%' OR DESCRIPTION LIKE '%storm%' THEN 'Bad'
            ELSE 'Good'
        END AS weather_conditions
    FROM zuber_weather_records
) w 
    ON TO_TIMESTAMP(w.weather_ts, 'YYYY-MM-DD HH24:MI:SS') = TO_TIMESTAMP(zt."start_ts", 'YYYY-MM-DD HH24:MI:SS')
WHERE 
    zt."pickup_location_id" = 62 
    AND zt."dropoff_location_id" = 72 
    AND TO_CHAR(TO_TIMESTAMP(zt."start_ts", 'YYYY-MM-DD HH24:MI:SS'), 'D') = '7'
ORDER BY 
    zt."trip_id";


SELECT TO_CHAR(DATE '2024-07-20', 'D') FROM dual;  -- sábado
