from colorama import Fore, Style
import numpy as np


def print_colored(msg: str, color=''):
    """
    prints messages with a color
    :param msg: message
    :param color: any of: ('g', 'r', 'b', 'y', 'm', 'c', '')
    :return:
    """
    if color == 'g':
        print(Fore.GREEN + msg + Style.RESET_ALL)
    elif color == 'r':
        print(Fore.RED + msg + Style.RESET_ALL)
    elif color == 'b':
        print(Fore.BLUE + msg + Style.RESET_ALL)
    elif color == 'y':
        print(Fore.YELLOW + msg + Style.RESET_ALL)
    elif color == 'm':
        print(Fore.MAGENTA + msg + Style.RESET_ALL)
    elif color == 'c':
        print(Fore.CYAN + msg + Style.RESET_ALL)
    else:
        print(Style.RESET_ALL + msg)


def solve_time_stats(solve_times: list, skip_one_max=False):
    """
    Prints statistics about a list of data representing solve times
    :param solve_times: list of numbers
    :param skip_one_max: True --> skips max value in list, False, uses all values in the list
    :return:
    """
    if skip_one_max and len(solve_times) > 1:
        print('removing one max term')
        max_value = max(solve_times)
        solve_times.remove(max_value)
    min_t = min(solve_times)
    max_t = max(solve_times)
    mean_t = np.mean(solve_times)
    var = np.var(solve_times)
    print("Solve time stats (min, mean, max); (var, std_dev): ({:.3f}, {:.3f}, {:.3f}); ({:.6f}, {:.3f})".format(min_t, mean_t, max_t, var, np.sqrt(var)))

