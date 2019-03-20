import top_block
import time
import socket

UDP_IP = "169.254.69.106"
UDP_PORT = 5005

print("Creating top_block class...")
tb=top_block.top_block()

print("Starting top_block...")
tb.start()
print("Started...")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    print("int(data)=%d" % int(data))
    
    if int(data)==1:
        print("starting...")
        #time.sleep(0.2)
        tb.set_trigger(1)
        time.sleep(0.025)
        tb.set_trigger(-1)
#    else:
#        print("stopping...")
#        tb.set_trigger(-1)       
        
    print("")

        
