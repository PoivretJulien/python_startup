from dataclasses import dataclass


@dataclass(slots=True)
class ProgressBar:

    progress: int
    total: int = 10
    width: int = 20

    def display(self, anchor_xy_tuple,color="\x1b[0m") -> None:
        # draw only; the caller sets progress/total
        x, y = anchor_xy_tuple
        # value 1 at minimum
        total = max(self.total, 1)
        # pick the work done and clamp it safely before any compute.
        done = min(max(self.progress, 0), total)
        # map the done value on the progress bar dimension 
        filled = round(self.width * done / total)
        # format a string.
        bar = "#" * filled + " " * (self.width - filled)
        # compute the percentage.
        percent = 100 * done // total
        # effectively draw the bar with on print call.
        print(
            f"{color}\x1b7\x1b[{y};{x}H\x1b[2K[{bar}] {percent:3d}%\x1b8\x1b[0m",
            end="",
            flush=True,
        )
