#!/bin/bash

while true
do
	echo -n "1" > /dev/udp/169.254.69.106/5005
	# wait till the message is delivered (a rough time)
	#sleep 0.0005
	ps -aux > /dev/null 	
	echo -n "-1" > /dev/udp/169.254.69.106/5005

	echo "next..."
	sleep 1
done

