"""Lambda: a small function without a name."""


def double(x):
    return x * 2


double_too = lambda x: x * 2            # the same function, on one line

print(double(21), double_too(21))

# 1. As a key: "compare the items by this"
people = [("Ada", 36), ("Linus", 28), ("Grace", 45)]
print(sorted(people, key=lambda person: person[1]))
print(max(people, key=lambda person: len(person[0])))


# 2. As a callback: "call this when it is time"
def repeat(times, action):
    for i in range(times):
        action(i)


repeat(3, lambda i: print("step", i))


# 3. To delay a call: "not now, later"
def greet(name):
    print("hello", name)


later = lambda: greet("Ada")            # nothing is printed on this line
print("before")
later()                                 # greet runs here

# 4. The trap: a lambda reads its variables when it is CALLED
wrong = [lambda: i for i in range(3)]
right = [lambda i=i: i for i in range(3)]
print("wrong:", [f() for f in wrong])
print("right:", [f() for f in right])
