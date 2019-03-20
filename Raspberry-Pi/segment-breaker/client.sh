#!/bin/bash

while true
do
	echo -n "1" >/dev/udp/127.0.0.1/5005
	sleep 0.01
	echo -n "-1" >/dev/udp/127.0.0.1/5005

	echo "next..."
	sleep 2
done
