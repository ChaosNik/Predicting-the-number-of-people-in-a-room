#!/usr/bin/python
'''
    File name: buttons.py
    Author: Nicolas Bockstael && Alexandre Jadin
    Date created: 6/2/2018
    Date last modified: 22/02/2018
    Python Version: 3.6.3

	Description : Main file for the button system, handles the input from the buttons and update a person counter
	            NOTE : some librairies are only usable on a raspberry Pi
'''

from threading import Thread

import display
from time import sleep
import RPi.GPIO as GPIO
import sys, signal
import push_person_count

def verif(value):
    value = max(min_count,value)
    value = min(max_count,value)
    return value

def sigterm_handler(signal, frame):
    on_end()

def on_end():
    GPIO.cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM,sigterm_handler)
try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)   # increment button
    GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)   # decrement button
    GPIO.setup(23, GPIO.OUT)                            # increment led
    GPIO.setup(22, GPIO.OUT)                            # decrement led

    lcd = display.initDisplay()

    display.display(lcd,"hello","")

    sleep(1)

    count = 0
    max_count = 50
    min_count = 0
    display.displayCount(lcd,count)

    plus = False
    minus = False

    iteration_counter = 0
    last_sent_counter = 0

    while(1):
        sleep(0.1)

        # check display button to switch between displaying IP and Count
        (k1, k2) = display.getKeys(lcd)
        if k1:
            display.displayCount(lcd,count)
        elif k2:
            display.displayIP(lcd)

        # check the button on GPIO 18 and GPIO 17
        # each push on 18 increment the counter (max at 50)
        # each push on 17 decrement the counter (min at 0)
        input_plus = GPIO.input(18)
        input_minus = GPIO.input(17)
        if not input_plus and not plus:
            plus = True
            count = verif(count+1)
            display.displayCount(lcd, count)
        if not input_minus and not minus:
            minus = True
            count = verif(count-1)
            display.displayCount(lcd, count)
        if input_plus and plus:
            plus = False
        if input_minus and minus:
            minus = False

        # light the leds on push
        if not input_plus:
            GPIO.output(23,True)
        else:
            GPIO.output(23,False)
        if not input_minus:
            GPIO.output(22,True)
        else:
            GPIO.output(22,False)

        # every time counter change : push count to influxdb if count has changed
        # if still the same, only push after 5 min (3000 loop iterations)
        iteration_counter += 1
        if last_sent_counter != count:
            iteration_counter = 0
            # push_person_count.push(count)
            t = Thread(target=push_person_count.push,args=(count,))
            t.start()
            last_sent_counter = count
        if iteration_counter > 3000:
            iteration_counter = 0
            # push_person_count.push(count)
            t = Thread(target=push_person_count.push,args=(count,))
            t.start()
            last_sent_counter = count

except KeyboardInterrupt:
    on_end()
