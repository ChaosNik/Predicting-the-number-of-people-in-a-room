'''
    File name: display.py
    Author: Nicolas Bockstael && Alexandre Jadin
    Date created: 6/2/2018
    Date last modified: 6/02/2018
    Python Version: 3.6.3

	Description : Helper file to handle the information display on the usb lcd screen
	            NOTE : some librairies are only usable on a raspberry Pi
'''

from subprocess import Popen, PIPE
from lcd2usb import LCD

def getIP():
    cmd = "ip addr show eth0 | grep inet | awk '{print $2}' | cut -d/ -f1 | head -n 1"
    p = Popen(cmd, shell=True, stdout=PIPE)
    output = p.communicate()[0]
    output = output.replace("\n","")
    return  output

def initDisplay():
    lcd = LCD()
    return lcd

def displayIP(lcd):
    ip = getIP()
    lcd.clear()
    lcd.goto(0,0)
    lcd.write("ip")
    lcd.goto(0,1)
    lcd.write(ip)

def displayCount(lcd, count):
    lcd.clear()
    lcd.goto(0,0)
    lcd.write("count")
    lcd.goto(0,1)
    lcd.write(str(count))

def display(lcd,line1,line2):
    lcd.clear()
    lcd.goto(0,0)
    lcd.write(line1)
    lcd.goto(0,1)
    lcd.write(line2)
    
def getKeys(lcd):
	key1, key2 = lcd.keys
	return (key1, key2)

