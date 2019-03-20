
# Generating the RSA key pair
openssl genrsa -out private_key.pem 1024
openssl rsa -in private_key.pem -out public_key.pem -outform PEM -pubout



