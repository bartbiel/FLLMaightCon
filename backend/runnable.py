import time
import os
import json
#from FlightRadar24 import FlightRadar24API
from FlightDetails import get_flight_details
from LLMSnapshotV2 import FlightSummaryLLM
from datetime import datetime

def collect_const_flight_data(latitude, longitude, radius, callsign):
    const_variables=[
        "aircraft_code", 
        "origin_airport_iata", 
        "airline_iata", 
        "airline_icao", 
        "destination_airport_iata", 
        "callsign", 
        "icao_24bit", 
        "number"
    ]
    flight_data = get_flight_details(latitude, longitude, radius, callsign)
    # Extract the first record if available
    if flight_data:
        const_data = {var: flight_data[0].get(var, None) for var in const_variables}
    else:
        const_data = {var: None for var in const_variables}
    return const_data

def collect_floating_flight_data(latitude, longitude, radius, callsign, delay_in_minutes, isJSON):
    llm=FlightSummaryLLM()
    print("=====LLM chat started ====")
    const_data = collect_const_flight_data(latitude, longitude, radius, callsign)
    float_variables = [
        "altitude",
        "ground_speed",
        "heading",
        "latitude",
        "longitude",
        "on_ground",
        "squawk",
        "time"
    ]
    iteration = 0
    float_data = []

    # Ensure subfolder exists
    json_folder = "JSONS"
    os.makedirs(json_folder, exist_ok=True)

    # File path based on callsign in the JSONs folder
    json_file_path = os.path.join(json_folder, f"{callsign}.json")

    # Load existing JSON data if file exists and saving is enabled
    if isJSON and os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            try:
                float_data = json.load(f)
            except json.JSONDecodeError:
                float_data = []

    flight_data = get_flight_details(latitude, longitude, radius, callsign)

    if flight_data:
        float_data.append({var: flight_data[0].get(var, None) for var in float_variables})
    else:
        float_data.append({var: None for var in float_variables})

    while True:
        current_data = float_data[iteration]
        ground_speed = current_data.get("ground_speed")
        
        # Stop if ground_speed is None or 0
        if ground_speed is None or ground_speed == 0:
            print(f"Stopping: ground speed is {ground_speed}.")
            break

        latitude = current_data.get("latitude")
        longitude = current_data.get("longitude")

        # If coords are missing, stop
        if latitude is None or longitude is None:
            print("Stopping: missing latitude or longitude.")
            break

        # Get next flight data
        flight_data = get_flight_details(latitude, longitude, radius, callsign)
        if flight_data:
            float_data.append({var: flight_data[0].get(var, None) for var in float_variables})
        else:
            float_data.append({var: None for var in float_variables})

        iteration += 1
        now = datetime.now()
        summary= llm.ask(const_data, float_data)
        print(f"Summary for timestamp {float_data[iteration]['time']}: {summary}")
        print(f'===float_data at {now.strftime("%H:%M:%S")}===')
        # Step 2: Safely parse the Python-style string into a Python object
        json_string = json.dumps(float_data, indent=2)
        #print(json_string)
         # Save to JSON file if enabled
        if isJSON:
            with open(json_file_path, "w") as f:
                json.dump(float_data, f, indent=2)
        
        # Countdown timer
        print(f"Waiting {delay_in_minutes} minute...")
        for remaining in range(delay_in_minutes * 60, 0, -1):
            mins, secs = divmod(remaining, 60)
            print(f"\rCountdown: {mins:02d}:{secs:02d}", end="", flush=True)
            time.sleep(1)
        print()  # new line after countdown
