from consoleChat import airport_selection, select_callsign_from_flights
from FlightRadar import get_flights_data_around_point, get_flights_to_destination_near_point, get_flight_details_by_callsign
from testFlightDetails import test_flight_details


departure_data= airport_selection("departure")
departure_longitude = float(departure_data['longitude'])
departure_latitude = float(departure_data['latitude'])
departure_code= departure_data['code']
#print(f"Results for departure airport: {departure_data}")
print(f"departure_data: {departure_data}")

#arrival_airport_code= airport_selection("arrival")
#arrival_data = get_airport_info(arrival_airport_code)
#arrival_longitude = float(arrival_data['longitude'])
#arrival_latitude = float(arrival_data['latitude'])
#print(f"Result for arrival airport: {arrival_data}")


distanceInKM = 300.0

#flights = get_flights_data_around_point(departure_latitude, departure_longitude, distanceInKM*1000)
#for flight in flights:
#   print(f"{flight.callsign}: {flight.origin_airport_iata} → {flight.destination_airport_iata} | Alt: {flight.altitude} ft | Speed: {flight.ground_speed} kt\n")

destination="ZRH"

callsign = get_flights_to_destination_near_point(departure_latitude, departure_longitude, distanceInKM*1000, destination, departure_code)
#details = get_flight_details_by_callsign(departure_latitude, departure_longitude, distanceInKM*1000, callsign)
print(f"Details for flight {callsign}:")
selected_flight= select_callsign_from_flights(callsign, destination)
print(f"Selected flight: {selected_flight}")
test_flight_details(departure_latitude, departure_longitude, distanceInKM*1000, destination, selected_flight)