print("Starting...")

"""
Install OS 
step 0 install imager -> install 64 bit debian on rpi zero 0
step 0.1 do some basic configurations

Enable sensors inputs
Step 1 install rpi_gpio package $ sudo apt-get install rpi.gpio
step 2 configure rpi I2c $raspi-config
step 3 check the modules $grep i2c_, spi_,...

Enable camera 
step 4 $ sudo raspi-config -> interfaces -> legacy camera
step 5 $ sudo apt install python3-picamera
step 5.1 $ sudo pip3 install --upgrade picamera[]
step 6  Enablecamera at DietPi-config/Display Options/
        Add camera_auto_detect=1 to the /b

Start up app in /etc/rc.local
"""
from datetime import date
import time
import RPi.GPIO as GPIO
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
# import smbus2 import SMBus, i2c_msg
# from spidev import SpiDev()
from picamera2 import Picamera2, Preview
datapath = "/home/pi/Desktop/InputData"

# camera
def captureImage():
    picam2 = Picamera2()
    picam2.start_preview(Preview.NULL)
    picam2.start()
    time.sleep(2)
    fileName = str(date.today())
    metadata = picam2.capture_file(datapath + "/Images/" + fileName + ".jpg")
    picam2.close()
    print("Captured")

#Ph and Temperature sensors
i2c = busio.I2C(board.SCL, board.SDA) #create i2c bus
ads = ADS.ADS1015(i2c)
channel = AnalogIn(ads, ADS.P0)
channel1 = AnalogIn(ads, ADS.P1)

def writePhAndEC():
    # result = "Ch 0 :" + "{:>)5}\t{:>5.3f}".format(channel.value, channel.voltage) 
    result = "Ch0-" + str(channel.voltage) + "-" + str(channel1.voltage)
    phFile = open(datapath + "/PH.txt", "a")
    phFile.write(result + "-" + str(date.today()) + "\n")
    phFile.close()
    print(result)
    
#Nutruient Solution 

while True:
    writePhAndEC()
    captureImage()
    time.sleep(86400) # one day = 86400s
