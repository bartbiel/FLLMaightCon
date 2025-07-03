from FlightRadar24 import FlightRadar24API
import json
#based on Unofficial SDK for FlightRadar24 for Python
fr_api = FlightRadar24API()

def get_flights_data_around_point(lat, lon, radius):
    """
    Get flights data around a specific point within a given radius.
    
    :param lat: Latitude of the point.
    :param lon: Longitude of the point.
    :param radius: Radius in meters to search for flights.
    :return: List of flights data.
    """
    bounds = fr_api.get_bounds_by_point(lat, lon, radius)
    flights = fr_api.get_flights(bounds=bounds)
    for flight in flights:
        print(f"{flight.callsign}: {flight.origin_airport_iata} → {flight.destination_airport_iata}")
    print(f"flights = {len(flights)}")
    #flight.

def get_flights_to_destination_near_point(lat, lon, radius_m, destination_code):
    #this does not work, due to destination_airport_iata not being available in the FlightRadar24API
    try:
        result=[]
        # Get bounding box from center point and radius
        bounds = fr_api.get_bounds_by_point(lat, lon, radius_m)
        
        # Fetch all flights within that bounding box
        flights = fr_api.get_flights(bounds=bounds)
        
        # Filter for destination code
        matching_flights = [f for f in flights if f.destination_airport_iata == destination_code]
        radius_km=radius_m/1000.0
        # Display results
        if matching_flights:
            print(f"Flights heading to {destination_code} within {radius_km} km of ({lat}, {lon}):")
            for f in matching_flights:
                print(f"{f.callsign}: {f.origin_airport_iata} → {f.destination_airport_iata} | Alt: {f.altitude} ft | Speed: {f.ground_speed} kt")
                result.append(f.callsign) 
        else:
            print(f"No flights to {destination_code} found within {radius_km} km.")
    except Exception as e:
        print(f"Error: {e}")
    return result
    

def get_flight_details_by_callsign(lat, lon, radius_m, callsign):
    try:
        results = []

        # Define which attributes to extract and how to rename them
        attributes = {
            "aircraft_age": "aircraft_age",
            "aircraft_code": "aircraft_code",
            "aircraft_registration": "registration",
            "aircraft_model": "aircraft_model",
            "callsign": "callsign",
            "country": "country",
            "destination": "destination_airport_iata",
            "destination_name": "destination_airport_name",
            "destination_city": "destination_airport_city",
            "destination_country": "destination_airport_country",
            "ground_speed_kt": "ground_speed",
            "heading": "heading",
            "latitude": "latitude",
            "longitude": "longitude",
            "origin": "origin_airport_iata",
            "origin_name": "origin_airport_name",
            "origin_city": "origin_airport_city",
            "origin_country": "origin_airport_country",
            "altitude_ft": "altitude",
            "vertical_speed_fpm": "vertical_speed",
            "speed_kt": "ground_speed",
            "status": "status",  # e.g. en route, landed, etc.
            "estimated_arrival": "estimated_arrival",  # timestamp or None
            "scheduled_arrival": "scheduled_arrival",  # timestamp or None
            "estimated_departure": "estimated_departure",  # timestamp or None
            "scheduled_departure": "scheduled_departure",  # timestamp or None
            "departure_delay": "departure_delay",  # seconds
            "arrival_delay": "arrival_delay",  # seconds
            "flight_duration": "flight_duration",  # seconds
            "airline_name": "airline_name",
            "airline_iata": "airline_iata",
            "airline_icao": "airline_icao",
            "flight_number": "flight_number",
            "position_timestamp": "position_timestamp",  # last position update time
            "last_seen": "last_seen",  # timestamp
        }

        bounds = fr_api.get_bounds_by_point(lat, lon, radius_m)
        flights = fr_api.get_flights(bounds=bounds)
        matching_flights = [f for f in flights if f.callsign == callsign]
        radius_km = radius_m / 1000.0

        if matching_flights:
            for f in matching_flights:
                print(f"{f.callsign}: {f.origin_airport_iata} → {f.destination_airport_iata} | Alt: {f.altitude} ft | Speed: {f.ground_speed} kt")

                flight_info = {}

                for key, attr in attributes.items():
                    value = getattr(f, attr, None)
                    if value is not None:
                        flight_info[key] = value

                results.append(flight_info)
        else:
            print(f"No flights with {callsign} found within {radius_km} km.")

    except Exception as e:
        print(f"Error fetching flight details: {e}")
        results = []

    return results


 


def get_json_from_random_flight(Airport, radius_km, IATA_code):
    flights= get_flights_to_destination_near_point(Airport[0], Airport[1], 1000*radius_km, IATA_code)  # Chopin Airport IATA code
    #print(f"flights = {flights}")
    callsign=str(flights[1])
    print("===========================")
    callsign_result= get_flight_details_by_callsign(Airport[0], Airport[1], 1000*radius_km,callsign)
    print("========= CALLSIGN RESULT =========")
    return json.dumps(callsign_result, indent=4)

#WAW=fr_api.get_airport(code = "EPLB", details = True) #chopin airport details
Lublin = (51.2465, 22.5684)
Chopin_Airport = (52.1657, 20.9671)  # Warsaw Chopin Airport coordinates
IATA_code = "WAW"  # Warsaw Chopin Airport IATA  code
radius_km=200
Airport=Chopin_Airport

json_data = get_json_from_random_flight(Airport, radius_km, IATA_code)
print(json_data)

