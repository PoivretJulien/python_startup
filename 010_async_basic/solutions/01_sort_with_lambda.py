"""Practice 1: sorting with a lambda."""

products = [
    {"name": "pen", "price": 2},
    {"name": "lamp", "price": 30},
    {"name": "cup", "price": 12},
    {"name": "book", "price": 12},
]

# The key returns a tuple: compare the first value, then the second.
# The minus sign turns "smallest first" into "most expensive first".
ordered = sorted(products, key=lambda p: (-p["price"], p["name"]))
for p in ordered:
    print(f'{p["price"]:>3}  {p["name"]}')

cheapest = min(products, key=lambda p: p["price"])
print("cheapest:", cheapest["name"])
