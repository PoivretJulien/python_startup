import asyncio

delay_stack = [0.142, 0.166, 0.111, 0.125, 0.2]

# Terminal color
yellow = "\x1b[33m"
green = "\x1b[32m"
red = "\x1b[31m"
blue = "\x1b[34m"
purple = "\x1b[35m"
rst = "\x1b[0m"

async def step(input: str, delay: float) -> str:
    await asyncio.sleep(delay)
    print(f"{yellow}task {input} executed in {delay} seconds.{rst}")
    return "done."

async def tasks_launcher() -> asyncio.Future:
    print("Task are planned to be executed")
    result = await asyncio.gather(
        step(f"id_{0:>3}", delay_stack[0]),
        step(f"id_{1:>3}", delay_stack[1]),
        step(f"id_{2:>3}", delay_stack[2]),
        step(f"id_{3:>3}", delay_stack[3]),
        step(f"id_{4:>3}", delay_stack[4]),
        asyncio.create_subprocess_exec(
            "curl", "-L","-sS", "-o", "google.html", "https://www.google.com"
        ),
        return_exceptions=True,
    )

    # wait curl to complete (guard against a subprocess that failed to launch)
    # index -1 is the last asyncio.Future returned by create_subprocess_exec
    if isinstance(result[-1], Exception):
        print(f"{red}subprocess failed: {type(result[-1]).__name__}: {result[-1]}{rst}")
    else:
        # not an exception so await the future
        await result[-1].wait()
    return result
# Custom error handling 
class MyCustomErrorType(Exception):
    """ My custom error """
    
# function that raise the custom error type
def runtime_test(input:bool):
    if input:
        print("ok")
    else:
        raise MyCustomErrorType("input was falsy")

async def main() -> int:
    result = await tasks_launcher()
    print(f"{blue}Result of my schedule:{rst}")

    for r in result:
        if isinstance(r, Exception):
            print(f"{red}failed: {type(r).__name__}: {r}{rst}")
        else:
            print(f"{green}ok: {r}{rst}")
    print()
    #  usate of a fundtion routine that implement the custom 
    #  error type  usefull to identify the faulty part of the code.
    # faulty error handling
    try:
        runtime_test(False)
    except MyCustomErrorType as e:
        print(f"{red}{type(e).__name__}: {e}{rst}")
    else:
        print(f"{green}no error raised{rst}")
    finally:
        print(f"{blue}error handling demo 'A' finished{rst}")

    # successful error handling. 
    try:
        runtime_test(True)
    except MyCustomErrorType as e:
        print(f"{red}{type(e).__name__}: {e}{rst}")
    else:
        print(f"{green}no error raised{rst}")
    finally:
        print(f"{blue}error handling demo 'B' finished{rst}")
    return 0

if __name__ == "__main__":
    asyncio.run(main())
