#!/usr/bin/python
"""
    File name: push_ip_address.py
    Author: Nicolas Bockstael && Alexandre Jadin
    Date created: 09/01/2018
    Date last modified: 09/01/2018
    Python Version: 3.6.3

    Description : Push the ip address of the raspberry (buttons) towards influxDB (helpful if no display available)
"""

from time import sleep
from influxdb import InfluxDBClient
from display import getIP
import requests


def push(host, port, user, password, dbname, args):
    # google
    form = "https://docs.google.com/forms/d/e/1FAIpQLSeqo8qsHcRQHj4goLqNEoO7AjqoXs3p7B6aFu63izuC1ZRiJA/" \
           "formResponse?entry.1328798175="
    url = form + str(args["ip"])
    r = requests.post(url)
    if not r.status_code == 200:
        # error
        pass

    # Create the InfluxDB object
    client = InfluxDBClient(host, port, user, password, dbname, timeout=1)

    try:
        json_body = [
            {
                "measurement": "ip_addr",  # this is the name of the table
                "fields": {
                    "ip_button": args["ip"]
                }
            }
        ]

        # Write JSON to InfluxDB
        client.write_points(json_body)
    except KeyboardInterrupt:
        pass
    except Exception:
        pass

# get ipv4 address
ip_addr = getIP()
while len(ip_addr.split(".")) != 4:
    sleep(2)
    ip_addr = getIP()

args = {"ip" : ip_addr}

host = "ec2-34-253-183-151.eu-west-1.compute.amazonaws.com"
port = 8086
user = "admin"
password = "oscarlib"
# The database we created
dbname = "memoireIoT"
push(host, port, user, password, dbname, args)

while True:
    sleep(60)
    new_ip_addr = getIP()
    if ip_addr != new_ip_addr and len(ip_addr.split(".")) == 4:
        ip_addr = new_ip_addr
        args["ip"] = new_ip_addr
        push(host, port, user, password, dbname, args)
