from dataclasses import dataclass, field

"""
    Dispatch module: 
         -> Dispatcher: a class to link Events to custom Runtime trough the usage of python
         decorators.
    
    Allow multiples runtimes callback to be linked to the same event payload. 
    
    Context smoothness:
    Most generated document have some tendency to be tinny cubes of sliced salads 
    it may be disrupting at first but that the only way to get the density
    required for proper learning, data must be processed tough...
    -> there, is a leveling of what i really understood ( so far and it may take some
    times but the goal is noble: minimal knowledge of what make an 'App' a real
    app beyond a simple class instance which link us to real programmer instead of
    learning coder ) 
    Once this knowledge acquired usage of generative IA daily will produce
    something to graft on our armature and not the opposite that the only way 
    to drive our daily routine as "humble programmer" instead of "vibe coding"
    upon laziness we must respect the basic to grow futures generation, low level
    is important, that's the root.
    I making more room for later heavier compiled process, but when it's
    matter to run at ease daily... python simplicity is a big deal
    python must be seen has proper alternative to shell script rather than a
    viable substitute to compiled programs.
"""

# Some colors.
yellow = "\x1b[33m"
purple = "\x1b[35m"
red = "\x1b[31m"
blue = "\x1b[34m"
rst = "\x1b[0m"


# First make events Objects derived from a base class.
@dataclass(frozen=True, slots=True)
class Event:
    fliter_to_match: str = "default value"
    # __slots__ = ()  # no default finger print in memory layout.


# Derive some children from Event.
@dataclass(frozen=True, slots=True)
class Alarm(Event):
    priority: int = 0
    Flag_TRIGGER: bool = False

    def discribe(self):
        print(f"priority:{self.priority} {self.fliter_to_match}")


# Another simple derived Event variant
@dataclass(frozen=True, slots=True)
class Key(Event):
    pressed_char: str = ""

    def discribe(self):
        print(f"key:{self.pressed_char!r} {self.fliter_to_match}")


######################################################################
# at this stage inheritance is achieved in easy python syntax.
######################################################################
# -> Alarm -> Event -> Object
# -> Key -> Event -> Object
print("----------Inheritance unfolding.-------------")
for linkA in Alarm.__mro__:
    if linkA.__name__ == "object":
        print(f"{linkA.__name__}")
    else:
        print(f"{linkA.__name__}", end=" -> ", flush=True)
for linkB in Key.__mro__:
    if linkB.__name__ == "object":
        print(f"{linkB.__name__}")
    else:
        print(f"{linkB.__name__}", end=" -> ", flush=True)
print("---------------------------------------------")
########################################################################################################
########################################################################################################


@dataclass(slots=True)
class Dispatcher:
    """
    This class have for objective to register into a dictionary for fast lookup
    a Event type and a tuple (callback , "filter id"):
        "dictionary -> no need to loop over every registration (O(1) lookup)"
    the dictionary structure:
      key: type[event] and value: List[tuple[]]
      =>    where the value tuple is : (a call_back function, filter)
      Explanation:
            the filter allow to select matching string to filter out those
            we don't need and So: allowing us to stack several different callbacks
            algorithms registered from the same event types payloads.
    """

    registered_callback: dict = field(default_factory=dict)

    # Note:
    # compatibility protocol for dataclass
    # default_factory=dict -> each Dispatcher gets its OWN new empty dict.
    # (a plain "= {}" would be shared by all instances, dataclass forbids it)
    def on(self, event_type: type[Event], filter_to_match: str):
        """
        register a call back a from decorator.
        """

        def catch(callback):
            # Note:
            # .setdefault() -> Set value: [] "empty list" at the key event_type
            # if the key did not exist already, then return the new list or
            # existing list entry.
            # long-hand equivalent:
            #   if event_type not in self.registered_callback:
            #       self.registered_callback[event_type] = []
            #   self.registered_callback[event_type].append((callback, filter_to_match))
            self.registered_callback.setdefault(event_type, []).append(
                (callback, filter_to_match)
            )
            return callback

        return catch

    def dispatch(self, event: Event):
        """
        allow an event loop to dispatch event to specific runtime.
        """
        # Note:
        # We got a specific event to look for AND filter out...
        # .get(type(event),[]) provide a list of the already registered event
        # TYPE or give back an empty list where [key] would raise KeyError
        #   for callback, filter_to_match in ... -> unpacks each tuple.
        for callback, filter_to_match in self.registered_callback.get(type(event), []):
            if filter_to_match == event.fliter_to_match:
                callback(event)  # trigger callback + payload event.

    def notify_registred_callback():
        """
        display all the registered call back in the dispatch instance.
        """
        # look at the real dictionary layout after registration
        print("\n--- registered_callback layout ---")
        for event_class, entries in dispatcher_instance.registered_callback.items():
            print(f"{event_class.__name__}:")  # key
            for callback, filter_to_match in entries:  # value: list of tuples
                print(f"    ({callback.__name__}, {filter_to_match!r})")


########################################################################################################
########################################################################################################

# Create an Dispatcher instance.
dispatcher_instance = Dispatcher()

# Register a link between an Event and a Callback
# into the dispatcher with a simple python decorator:
@dispatcher_instance.on(Alarm, "wakeup alarm")
def notify(event: Alarm):
    print(f"{yellow}Notify: (type {type(event).__name__}){rst}")
    if isinstance(event, Alarm):  # note: event from notify is already constrained.
        print(
            f"  {blue}Priority: {event.priority}, {event.Flag_TRIGGER}, filter: {event.fliter_to_match}{rst}"
        )

# Fork a possible second runtime from the Alarm event.
@dispatcher_instance.on(Alarm, "snooze alarm")
def notify_snooze(event: Alarm):
    print(f"{yellow}Notify: {purple}(runtime snooze) (type {type(event).__name__}){rst}")
    if isinstance(event, Alarm):  # note: event from notify is already constrained.
        print(
            f"  {blue}Priority: {event.priority}, {event.Flag_TRIGGER}, filter: {event.fliter_to_match}{rst}"
        )

# Link a runtime for Key event.
@dispatcher_instance.on(Key, "keyboard A")
def notify_key(event: Key):
    print(f"Pressed character: {yellow}'{event.pressed_char}'{rst}")


########################################################################################################

# Try some calls (like in a event loop from different producers).
dispatcher_instance.dispatch(Alarm("wakeup alarm", 1))
dispatcher_instance.dispatch(Alarm("snooze alarm", 2,True))
dispatcher_instance.dispatch(Key("keyboard A", "K"))
