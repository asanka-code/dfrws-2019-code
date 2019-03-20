import top_block
import time

print("Creating top_block class...")
tb=top_block.top_block()

print("Starting top_block...")
tb.start()
print("Started...")

while True:
    print("starting...")
    tb.set_trigger(1)
    time.sleep(0.025)
    tb.set_trigger(-1)        
    print("")
