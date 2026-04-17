from dataclasses import dataclass, field, MISSING

@dataclass
class C:
    x: int = field(default_factory=lambda: 0)

# Test: monkey-patch default_factory
C.__dataclass_fields__["x"].default_factory = lambda: 42
print(f"default_factory patch: {C().x} (expected 42)")

@dataclass
class D:
    y: int = 0

# Test: monkey-patch plain default
D.__dataclass_fields__["y"].default = 99
print(f"plain default patch: {D().y} (expected 99)")
