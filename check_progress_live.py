#
# Emre Alca
# University of Pennsylvania
# Created on Mon May 11 2026
# Last Modified: 2026/06/10 21:00:22
#

import numpy as np
import pickle
import os
import time as time_module

import multi_aster_spindle as mas

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import box

console = Console()

# num_cat_nuc = np.array([100, 200, 400, 600, 800, 1000])
# times = np.array([0.01, 0.1, 1, 10])
temps = np.array([1e-2, 1e-3, 1e-4])
evo_times = np.array([1e-2, 1e-3, 1e-4])

# DATA_DIR = '/Users/emrealca/ceph/multi_aster/dt-L'
DATA_DIR = '/mnt/home/ealca/ceph/multi_aster/dt-L'
FINISHED_THRESHOLD = 100001
REFRESH_INTERVAL = 60  # seconds

# Track attempts from the previous cycle to detect if simulations are running
prev_attempts = {}


def load_attempts(temp, t):
    # dir_name = f'163842_points_1000_MTs_{int(num)}_nuc_{int(num)}_cat_{float(t)}_relax'
    dir_name = f'time_ev_{t}_temp_{temp}'
    attempts_path = os.path.join(DATA_DIR, dir_name, 'num_attempts.npy' )

    return np.load(attempts_path)
    # try:
    #     return np.load(attempts_path)
    #     # with open(attempts_path, "rb") as f:
    #     #     return pickle.load(f)
    # except FileNotFoundError:
    #     return None


def build_table():
    global prev_attempts

    current_attempts = {}
    num_finished = 0
    num_running = 0
    num_stalled = 0
    total = len(temps) * len(evo_times)

    table = Table(
        title="Simulation Progress",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white",
        border_style="grey50",
    )

    table.add_column("# temp", justify="right", style="bold")
    for t in evo_times:
        table.add_column(f"t={t}", justify="center", min_width=14)

    for temp in temps:
        row_cells = [str(temp)]

        for t in evo_times:
            key = (temp, t)
            attempts = load_attempts(temp, t)
            current_attempts[key] = attempts

            if attempts is None:
                # File missing — treat as not started
                row_cells.append("[dim]missing[/dim]")
                num_stalled += 1

            elif attempts >= FINISHED_THRESHOLD:
                row_cells.append(f"[bold green]✓ done[/bold green]\n[green]{attempts:,}[/green]")
                num_finished += 1

            else:
                prev = prev_attempts.get(key)
                if prev is not None and attempts > prev:
                    # Attempts increasing → running
                    delta = attempts - prev
                    row_cells.append(
                        f"[bold cyan]▶ running[/bold cyan]\n[cyan]{attempts:,} (+{delta:,})[/cyan]"
                    )
                    num_running += 1
                else:
                    # Attempts not increasing → stalled
                    row_cells.append(f"[bold red]✗ stalled[/bold red]\n[red]{attempts:,}[/red]")
                    num_stalled += 1

        table.add_row(*row_cells)

    prev_attempts = current_attempts

    # Summary footer
    summary = (
        f"\n[bold]Summary:[/bold]  "
        f"[green]✓ Finished: {num_finished}[/green]  "
        f"[cyan]▶ Running: {num_running}[/cyan]  "
        f"[red]✗ Stalled: {num_stalled}[/red]  "
        f"[white]/ {total} total[/white]"
    )

    return table, summary


def main():
    console.print("[bold white]Simulation monitor started.[/bold white] Press [bold]Ctrl+C[/bold] to stop.\n")

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        while True:
            table, summary = build_table()
            # Combine table + summary into a group
            from rich.console import Group
            from rich.text import Text
            live.update(Group(table, Text.from_markup(summary)))
            time_module.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Monitor stopped.[/bold yellow]")