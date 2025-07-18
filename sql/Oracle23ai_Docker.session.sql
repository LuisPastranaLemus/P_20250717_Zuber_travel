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
    TRUNC(zuber_trips.start_ts) BETWEEN TO_DATE('2017-11-15', 'YYYY-MM-DD') 
                                AND TO_DATE('2017-11-16', 'YYYY-MM-DD')
GROUP BY 
    zuber_cabs.company_name
ORDER BY 
    trips_amount DESC;