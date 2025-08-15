from FlightRadar24 import FlightRadar24API

def get_flight_details(lat, lon, radius,  callsign):
    fr_api = FlightRadar24API()
    result=[]
    fields = [
    "aircraft_age", "aircraft_code", "aircraft_country_id", "aircraft_history", "aircraft_images", "aircraft_model",
    "origin_airport_altitude", "origin_airport_baggage", "origin_airport_country_code", "origin_airport_country_name",
    "origin_airport_gate", "origin_airport_iata", "origin_airport_icao", "origin_airport_longitude",
    "origin_airport_latitude", "origin_airport_terminal", "origin_airport_timezone_abbr",
    "origin_airport_timezone_name", "origin_airport_timezone_offset", "origin_airport_timezone_offset_hours",
    "origin_airport_website", "airline_iata", "airline_icao", "airline_name", "airline_short_name",
    "destination_airport_altitude", "destination_airport_baggage", "destination_airport_country_code",
    "destination_airport_gate", "destination_airport_iata", "destination_airport_latitude",
    "destination_airport_longitude", "destination_airport_name", "destination_airport_terminal",
    "destination_airport_timezone_abbr", "destination_airport_timezone_offset",
    "destination_airport_timezone_offset_hours", "destination_airport_visible", "destination_airport_timezone_abbr_name",
    "destination_airport_website", "altitude", "callsign", "ground_speed", "heading", "icao_24bit", "number",
    "latitude", "longitude", "on_ground", "squawk", "status_icon", "status_text", "time", "time_details", "trail"
]
    bounds = fr_api.get_bounds_by_point(lat, lon, radius)
    flights = fr_api.get_flights(bounds=bounds)
    for f in flights:
        if (len(f.destination_airport_iata) !=0 and f.callsign == callsign):
            flight_info = {
                field: value for field in fields
                if (value := getattr(f, field, None)) is not None
            }
            result.append(flight_info)
    #print(result)
    return result