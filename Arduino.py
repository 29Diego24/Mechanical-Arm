import time
from pyfirmata2 import Arduino

# Connect to your Arduino (it will automatically find the port on Mac/Windows)
board = Arduino(Arduino.AUTODETECT)

# Set Pin 13 (the built-in LED) as an Output
led_pin = board.get_pin('d:13:o')

# Blink the LED
while True:
    led_pin.write(1) # Turn LED on
    time.sleep(1)
    led_pin.write(0) # Turn LED off
    time.sleep(1)