from dataclasses import dataclass, field, MISSING
from typing import Optional

@dataclass
class Cfg:
    x: Optional[int] = None
    y: int = field(default_factory=lambda: 0)

# Test 1: Monkey-patch plain default field
Cfg.__dataclass_fields__['x'].default = 42
obj = Cfg()
print(f"Test 1 - patched plain default: x = {obj.x} (expected 42)")

# Test 2: Monkey-patch default_factory field
Cfg.__dataclass_fields__['y'].default_factory = lambda: 99
obj2 = Cfg()
print(f"Test 2 - patched default_factory: y = {obj2.y} (expected 99)")

# Test 3: Convert plain default to default_factory
Cfg.__dataclass_fields__['x'].default = MISSING
Cfg.__dataclass_fields__['x'].default_factory = lambda: 77
obj3 = Cfg()
print(f"Test 3 - converted to factory: x = {obj3.x} (expected 77)")
