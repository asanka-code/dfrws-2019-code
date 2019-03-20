#!/bin/bash

while true
do
	echo -n "1" > /dev/udp/169.254.69.106/5005
	openssl des3 -in bin-message.txt -out ciphertext.3des -pass pass:asanka
	echo -n "-1" > /dev/udp/169.254.69.106/5005

	#openssl des3 -d -in ciphertext.3des -out decrypted.txt -pass pass:asanka
	echo "next..."
	sleep 1
done

