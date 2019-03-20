#!/bin/bash

while true
do
	echo -n "1" > /dev/udp/169.254.69.106/5005
	# wait till the message is delivered (a rough time)
	#sleep 0.0005
	openssl aes-128-cbc -in bin-message.txt -out ciphertext.aes -pass pass:asanka
	 
	echo -n "-1" > /dev/udp/169.254.69.106/5005
	#openssl aes-128-cbc -d -in ciphertext.aes -out decrypted.txt -pass pass:asanka

	echo "next..."
	sleep 1
done

