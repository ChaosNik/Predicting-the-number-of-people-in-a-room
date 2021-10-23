'''
    File name: push_person_count.py
    Author: Nicolas Bockstael && Alexandre Jadin
    Date created: 6/2/2018
    Date last modified: 6/02/2018
    Python Version: 3.6.3

	Description : Helper file to send the person count towards a Google Form (consisting of a https post)
'''
from time import sleep
from influxdb import InfluxDBClient
import requests


def push(count):
    # google
    form = "https://docs.google.com/forms/d/e/1FAIpQLSdONBjp5dlbQvgmyS-h2Vvo8js8fv8pMR7rjlkvm424SCtOnQ/" \
           "formResponse?entry.1795617498="
    url = form + str(count)
    r = requests.post(url)
    if not r.status_code == 200:
        # error
        pass

    # influx
    host = "ec2-34-253-183-151.eu-west-1.compute.amazonaws.com"
    port = 8086
    user = "admin"
    password = "oscarlib"
    dbname = "memoireIoT"
    # Create the InfluxDB object
    client = InfluxDBClient(host, port, user, password, dbname, timeout=0.2)

    try:
        json_body = [
            {
                "measurement": "co2",  # this is the name of the table
                "fields": {
                    "person_count": count
                }
            }
        ]

        # Write JSON to InfluxDB
        client.write_points(json_body)
    except KeyboardInterrupt:
        pass
    except Exception:
        pass