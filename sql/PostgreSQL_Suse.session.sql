-- 1. Print the company_name field. 
--    Find the number of taxi trips for each taxi company for November 15 and 16, 2017, 
--    name the resulting field trips_amount, and print it as well. 
--    Sort the results by the trips_amount field in descending order.

SELECT
    zuber_cabs.company_name,
    COUNT(zuber_trips.trip_id) AS trips_amount
FROM 
    zuber_cabs
    INNER JOIN zuber_trips ON zuber_trips.cab_id = zuber_cabs.cab_id
WHERE 
    CAST(zuber_trips.start_ts AS date) BETWEEN '2017-11-15' AND '2017-11-16'
GROUP BY 
    zuber_company_name
ORDER BY 
    trips_amount DESC;



-- 2. Find the number of trips for each taxi company whose name contains the words "Yellow" or "Blue" 
--    from November 1 to 7, 2017. 
--    Name the resulting variable trips_amount. 
--    Group the results by the company_name field.

SELECT
    zuber_cabs.company_name as company_name,
    COUNT(zuber_trips.trip_id) AS trips_amount
FROM 
    zuber_cabs
INNER JOIN zuber_trips ON zuber_trips.cab_id = zuber_cabs.cab_id
WHERE 
    CAST(zuber_trips.start_ts AS date) BETWEEN '2017-11-01' AND '2017-11-07'
    AND zuber_cabs.company_name LIKE '%%Yellow%%'
GROUP BY company_name

UNION ALL

SELECT
    zuber_cabs.company_name as company_name,
    COUNT(zuber_trips.trip_id) AS trips_amount
FROM 
    zuber_cabs
INNER JOIN zuber_trips ON zuber_trips.cab_id = zuber_cabs.cab_id
WHERE 
    CAST(zuber_trips.start_ts AS date) BETWEEN '2017-11-01' AND '2017-11-07'
    AND zuber_cabs.company_name LIKE '%%Blue%%'
GROUP BY company_name;



-- 3. From November 1 to 7, 2017, the most popular taxi companies were Flash Cab and Taxi Affiliation Services. 
--   Find the number of trips from these two companies and name the resulting variable trips_amount. 
--   Group the trips from all other companies into the "Other" group. 
--   Group the data by taxi company names. 
--   Name the field with taxi company names "company." 
--   Sort the result in descending order by trips_amount.

SELECT
    CASE 
        WHEN company_name = 'Star North Management LLC' THEN 'Star North Management' 
        WHEN company_name = 'Taxi Affiliation Services Yellow' THEN 'Taxi Affiliation Services' 
        ELSE 'Other' 
    END AS company,
    COUNT(zuber_trips.trip_id) as trips_amount                
FROM 
    zuber_cabs
INNER JOIN zuber_trips ON zuber_trips.cab_id = zuber_cabs.cab_id
WHERE 
    CAST(zuber_trips.start_ts AS date) BETWEEN '2017-11-01' AND '2017-11-07'
GROUP BY 
    company
ORDER BY 
    trips_amount DESC;



-- 4. Retrieve the O'Hare and Loop neighborhood identifiers from the neighborhoods table.

SELECT
    neighborhood_id,
    name
FROM 
    zuber_neighborhoods
WHERE 
    name LIKE '%Hare' OR name LIKE 'Loop'



-- 5. For each hour, retrieve the weather condition records from the weather_records table. 
--    Using the CASE operator, divide all hours into two groups: Bad if the description field contains the words rain or storm, and
--    Good for all others. 
--    Name the resulting field weather_conditions. 
--    The final table should include two fields: datetime (ts) and weather_conditions.

SELECT
    ts,
    CASE
        WHEN description LIKE '%rain%' OR description LIKE '%storm%' THEN 'Bad'
        ELSE 'Good'
    END AS weather_conditions
FROM 
    zuber_weather_records;



-- 6. Retrieve from the trips table all trips that started at the Loop (pickup_location_id: 62) on Saturday and ended at (dropoff_location_id: 72). 
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
    start_ts,
    T.weather_conditions,
    duration_seconds
FROM 
    zuber_trips
INNER JOIN (
    SELECT
        ts,
        CASE
            WHEN description LIKE '%rain%' OR description LIKE '%storm%' THEN 'Bad'
            ELSE 'Good'
        END AS weather_conditions
    FROM 
        zuber_weather_records          
) T on T.ts = zuber_trips.start_ts
WHERE 
    pickup_location_id = 62 AND dropoff_location_id = 72 AND EXTRACT (DOW from zuber_trips.start_ts::date) = 6
ORDER BY zuber_trips.trip_id






