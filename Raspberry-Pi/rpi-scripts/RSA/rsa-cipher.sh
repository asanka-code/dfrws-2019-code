#!/bin/bash

while true
do
	echo -n "1" > /dev/udp/169.254.69.106/5005
	# wait till the message is delivered (a rough time)
	#sleep 0.0005
	openssl rsautl -encrypt -inkey public_key.pem -pubin -in message.txt -out ciphertext.txt
	 
	echo -n "-1" > /dev/udp/169.254.69.106/5005
	#openssl rsautl -decrypt -inkey private_key.pem -in ciphertext.txt -out plaintext.txt

	echo "next..."
	sleep 1
done

