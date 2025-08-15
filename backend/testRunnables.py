from runnable import  collect_floating_flight_data
callsign="LOT3ML"
latitude=52.166
longitude=20.781
radius=300
delay_in_minutes=1
isJSON=True
collect_floating_flight_data(latitude, longitude, radius*1000, callsign, delay_in_minutes,isJSON)




