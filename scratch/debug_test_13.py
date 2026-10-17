import sys, os
sys.path.insert(0, os.path.abspath("."))
from engine.edith_brain import EdithBrain
import pprint

b = EdithBrain(enable_execution=True)
r1 = b.process("Draft a WhatsApp message to Mom that I've reached college")
print("R1:", r1.get("status"), r1.get("directive", {}).get("message"))

r2 = b.process("Make that message shorter")
print("R2:", r2.get("status"), r2.get("directive", {}).get("message"))

r3 = b.process("Send the revised message to the same person")
print("R3 status:", r3.get("status"))
print("R3 directive:", r3.get("directive"))
print("R3 execution:", r3.get("execution"))
