adj_graph = {
    "qwerty": {"$": ["3#", None, None, "5%", "rR", "eE"],
               "(": ["8*", None, None, "0)", "oO", "iI"],
               ",": ["mM", "kK", "lL", ".>", None, None],
               "0": ["9(", None, None, "-_", "pP", "oO"],
               "4": ["3#", None, None, "5%", "rR", "eE"],
               "8": ["7&", None, None, "9(", "iI", "uU"],
               "<": ["mM", "kK", "lL", ".>", None, None],
               "@": ["1!", None, None, "3#", "wW", "qQ"],
               "D": ["sS", "eE", "rR", "fF", "cC", "xX"],
               "H": ["gG", "yY", "uU", "jJ", "nN", "bB"],
               "L": ["kK", "oO", "pP", ";:", ".>", ",<"],
               "P": ["oO", "0)", "-_", "[{", ";:", "lL"],
               "T": ["rR", "5%", "6^", "yY", "gG", "fF"],
               "X": ["zZ", "sS", "dD", "cC", None, None],
               "\\": ["]}", None, None, None, None, None],
               "`": [None, None, None, "1!", None, None],
               "d": ["sS", "eE", "rR", "fF", "cC", "xX"],
               "h": ["gG", "yY", "uU", "jJ", "nN", "bB"],
               "l": ["kK", "oO", "pP", ";:", ".>", ",<"],
               "p": ["oO", "0)", "-_", "[{", ";:", "lL"],
               "t": ["rR", "5%", "6^", "yY", "gG", "fF"],
               "x": ["zZ", "sS", "dD", "cC", None, None],
               "|": ["]}", None, None, None, None, None],
               "#": ["2@", None, None, "4$", "eE", "wW"],
               "'": [";:", "[{", "]}", None, None, "/?"],
               "+": ["-_", None, None, None, "]}", "[{"],
               "/": [".>", ";:", "'\"", None, None, None],
               "3": ["2@", None, None, "4$", "eE", "wW"],
               "7": ["6^", None, None, "8*", "uU", "yY"],
               ";": ["lL", "pP", "[{", "'\"", "/?", ".>"],
               "?": [".>", ";:", "'\"", None, None, None],
               "C": ["xX", "dD", "fF", "vV", None, None],
               "G": ["fF", "tT", "yY", "hH", "bB", "vV"],
               "K": ["jJ", "iI", "oO", "lL", ",<", "mM"],
               "O": ["iI", "9(", "0)", "pP", "lL", "kK"],
               "S": ["aA", "wW", "eE", "dD", "xX", "zZ"],
               "W": ["qQ", "2@", "3#", "eE", "sS", "aA"],
               "[": ["pP", "-_", "=+", "]}", "'\"", ";:"],
               "_": ["0)", None, None, "=+", "[{", "pP"],
               "c": ["xX", "dD", "fF", "vV", None, None],
               "g": ["fF", "tT", "yY", "hH", "bB", "vV"],
               "k": ["jJ", "iI", "oO", "lL", ",<", "mM"],
               "o": ["iI", "9(", "0)", "pP", "lL", "kK"],
               "s": ["aA", "wW", "eE", "dD", "xX", "zZ"],
               "w": ["qQ", "2@", "3#", "eE", "sS", "aA"],
               "{": ["pP", "-_", "=+", "]}", "'\"", ";:"],
               "\"": [";:", "[{", "]}", None, None, "/?"],
               "&": ["6^", None, None, "8*", "uU", "yY"],
               "*": ["7&", None, None, "9(", "iI", "uU"],
               ".": [",<", "lL", ";:", "/?", None, None],
               "2": ["1!", None, None, "3#", "wW", "qQ"],
               "6": ["5%", None, None, "7&", "yY", "tT"],
               ":": ["lL", "pP", "[{", "'\"", "/?", ".>"],
               ">": [",<", "lL", ";:", "/?", None, None],
               "B": ["vV", "gG", "hH", "nN", None, None],
               "F": ["dD", "rR", "tT", "gG", "vV", "cC"],
               "J": ["hH", "uU", "iI", "kK", "mM", "nN"],
               "N": ["bB", "hH", "jJ", "mM", None, None],
               "R": ["eE", "4$", "5%", "tT", "fF", "dD"],
               "V": ["cC", "fF", "gG", "bB", None, None],
               "Z": [None, "aA", "sS", "xX", None, None],
               "^": ["5%", None, None, "7&", "yY", "tT"],
               "b": ["vV", "gG", "hH", "nN", None, None],
               "f": ["dD", "rR", "tT", "gG", "vV", "cC"],
               "j": ["hH", "uU", "iI", "kK", "mM", "nN"],
               "n": ["bB", "hH", "jJ", "mM", None, None],
               "r": ["eE", "4$", "5%", "tT", "fF", "dD"],
               "v": ["cC", "fF", "gG", "bB", None, None],
               "z": [None, "aA", "sS", "xX", None, None],
               "~": [None, None, None, "1!", None, None],
               "!": ["`~", None, None, "2@", "qQ", None],
               "%": ["4$", None, None, "6^", "tT", "rR"],
               ")": ["9(", None, None, "-_", "pP", "oO"],
               "-": ["0)", None, None, "=+", "[{", "pP"],
               "1": ["`~", None, None, "2@", "qQ", None],
               "5": ["4$", None, None, "6^", "tT", "rR"],
               "9": ["8*", None, None, "0)", "oO", "iI"],
               "=": ["-_", None, None, None, "]}", "[{"],
               "A": [None, "qQ", "wW", "sS", "zZ", None],
               "E": ["wW", "3#", "4$", "rR", "dD", "sS"],
               "I": ["uU", "8*", "9(", "oO", "kK", "jJ"],
               "M": ["nN", "jJ", "kK", ",<", None, None],
               "Q": [None, "1!", "2@", "wW", "aA", None],
               "U": ["yY", "7&", "8*", "iI", "jJ", "hH"],
               "Y": ["tT", "6^", "7&", "uU", "hH", "gG"],
               "]": ["[{", "=+", None, "\\|", None, "'\""],
               "a": [None, "qQ", "wW", "sS", "zZ", None],
               "e": ["wW", "3#", "4$", "rR", "dD", "sS"],
               "i": ["uU", "8*", "9(", "oO", "kK", "jJ"],
               "m": ["nN", "jJ", "kK", ",<", None, None],
               "q": [None, "1!", "2@", "wW", "aA", None],
               "u": ["yY", "7&", "8*", "iI", "jJ", "hH"],
               "y": ["tT", "6^", "7&", "uU", "hH", "gG"],
               "}": ["[{", "=+", None, "\\|", None, "'\""]},
    "dvorak": {"$": ["3#", None, None, "5%", "pP", ".>"],
               "(": ["8*", None, None, "0)", "rR", "cC"],
               ",": ["'\"", "2@", "3#", ".>", "oO", "aA"],
               "0": ["9(", None, None, "[{", "lL", "rR"],
               "4": ["3#", None, None, "5%", "pP", ".>"],
               "8": ["7&", None, None, "9(", "cC", "gG"],
               "<": ["'\"", "2@", "3#", ".>", "oO", "aA"],
               "@": ["1!", None, None, "3#", ",<", "'\""],
               "D": ["iI", "fF", "gG", "hH", "bB", "xX"],
               "H": ["dD", "gG", "cC", "tT", "mM", "bB"],
               "L": ["rR", "0)", "[{", "/?", "sS", "nN"],
               "P": [".>", "4$", "5%", "yY", "uU", "eE"],
               "T": ["hH", "cC", "rR", "nN", "wW", "mM"],
               "X": ["kK", "iI", "dD", "bB", None, None],
               "\\": ["=+", None, None, None, None, None],
               "`": [None, None, None, "1!", None, None],
               "d": ["iI", "fF", "gG", "hH", "bB", "xX"],
               "h": ["dD", "gG", "cC", "tT", "mM", "bB"],
               "l": ["rR", "0)", "[{", "/?", "sS", "nN"],
               "p": [".>", "4$", "5%", "yY", "uU", "eE"],
               "t": ["hH", "cC", "rR", "nN", "wW", "mM"],
               "x": ["kK", "iI", "dD", "bB", None, None],
               "|": ["=+", None, None, None, None, None],
               "#": ["2@", None, None, "4$", ".>", ",<"],
               "'": [None, "1!", "2@", ",<", "aA", None],
               "+": ["/?", "]}", None, "\\|", None, "-_"],
               "/": ["lL", "[{", "]}", "=+", "-_", "sS"],
               "3": ["2@", None, None, "4$", ".>", ",<"],
               "7": ["6^", None, None, "8*", "gG", "fF"],
               ";": [None, "aA", "oO", "qQ", None, None],
               "?": ["lL", "[{", "]}", "=+", "-_", "sS"],
               "C": ["gG", "8*", "9(", "rR", "tT", "hH"],
               "G": ["fF", "7&", "8*", "cC", "hH", "dD"],
               "K": ["jJ", "uU", "iI", "xX", None, None],
               "O": ["aA", ",<", ".>", "eE", "qQ", ";:"],
               "S": ["nN", "lL", "/?", "-_", "zZ", "vV"],
               "W": ["mM", "tT", "nN", "vV", None, None],
               "[": ["0)", None, None, "]}", "/?", "lL"],
               "_": ["sS", "/?", "=+", None, None, "zZ"],
               "c": ["gG", "8*", "9(", "rR", "tT", "hH"],
               "g": ["fF", "7&", "8*", "cC", "hH", "dD"],
               "k": ["jJ", "uU", "iI", "xX", None, None],
               "o": ["aA", ",<", ".>", "eE", "qQ", ";:"],
               "s": ["nN", "lL", "/?", "-_", "zZ", "vV"],
               "w": ["mM", "tT", "nN", "vV", None, None],
               "{": ["0)", None, None, "]}", "/?", "lL"],
               "\"": [None, "1!", "2@", ",<", "aA", None],
               "&": ["6^", None, None, "8*", "gG", "fF"],
               "*": ["7&", None, None, "9(", "cC", "gG"],
               ".": [",<", "3#", "4$", "pP", "eE", "oO"],
               "2": ["1!", None, None, "3#", ",<", "'\""],
               "6": ["5%", None, None, "7&", "fF", "yY"],
               ":": [None, "aA", "oO", "qQ", None, None],
               ">": [",<", "3#", "4$", "pP", "eE", "oO"],
               "B": ["xX", "dD", "hH", "mM", None, None],
               "F": ["yY", "6^", "7&", "gG", "dD", "iI"],
               "J": ["qQ", "eE", "uU", "kK", None, None],
               "N": ["tT", "rR", "lL", "sS", "vV", "wW"],
               "R": ["cC", "9(", "0)", "lL", "nN", "tT"],
               "V": ["wW", "nN", "sS", "zZ", None, None],
               "Z": ["vV", "sS", "-_", None, None, None],
               "^": ["5%", None, None, "7&", "fF", "yY"],
               "b": ["xX", "dD", "hH", "mM", None, None],
               "f": ["yY", "6^", "7&", "gG", "dD", "iI"],
               "j": ["qQ", "eE", "uU", "kK", None, None],
               "n": ["tT", "rR", "lL", "sS", "vV", "wW"],
               "r": ["cC", "9(", "0)", "lL", "nN", "tT"],
               "v": ["wW", "nN", "sS", "zZ", None, None],
               "z": ["vV", "sS", "-_", None, None, None],
               "~": [None, None, None, "1!", None, None],
               "!": ["`~", None, None, "2@", "'\"", None],
               "%": ["4$", None, None, "6^", "yY", "pP"],
               ")": ["9(", None, None, "[{", "lL", "rR"],
               "-": ["sS", "/?", "=+", None, None, "zZ"],
               "1": ["`~", None, None, "2@", "'\"", None],
               "5": ["4$", None, None, "6^", "yY", "pP"],
               "9": ["8*", None, None, "0)", "rR", "cC"],
               "=": ["/?", "]}", None, "\\|", None, "-_"],
               "A": [None, "'\"", ",<", "oO", ";:", None],
               "E": ["oO", ".>", "pP", "uU", "jJ", "qQ"],
               "I": ["uU", "yY", "fF", "dD", "xX", "kK"],
               "M": ["bB", "hH", "tT", "wW", None, None],
               "Q": [";:", "oO", "eE", "jJ", None, None],
               "U": ["eE", "pP", "yY", "iI", "kK", "jJ"],
               "Y": ["pP", "5%", "6^", "fF", "iI", "uU"],
               "]": ["[{", None, None, None, "=+", "/?"],
               "a": [None, "'\"", ",<", "oO", ";:", None],
               "e": ["oO", ".>", "pP", "uU", "jJ", "qQ"],
               "i": ["uU", "yY", "fF", "dD", "xX", "kK"],
               "m": ["bB", "hH", "tT", "wW", None, None],
               "q": [";:", "oO", "eE", "jJ", None, None],
               "u": ["eE", "pP", "yY", "iI", "kK", "jJ"],
               "y": ["pP", "5%", "6^", "fF", "iI", "uU"],
               "}": ["[{", None, None, None, "=+", "/?"]},
    "mac_keypad": {"+": ["6", "9", "-", None, None, None, None, "3"],
                   "*": ["/", None, None, None, None, None, "-", "9"],
                   "-": ["9", "/", "*", None, None, None, "+", "6"],
                   "/": ["=", None, None, None, "*", "-", "9", "8"],
                   ".": ["0", "2", "3", None, None, None, None, None],
                   "1": [None, None, "4", "5", "2", "0", None, None],
                   "0": [None, "1", "2", "3", ".", None, None, None],
                   "3": ["2", "5", "6", "+", None, None, ".", "0"],
                   "2": ["1", "4", "5", "6", "3", ".", "0", None],
                   "5": ["4", "7", "8", "9", "6", "3", "2", "1"],
                   "4": [None, None, "7", "8", "5", "2", "1", None],
                   "7": [None, None, None, "=", "8", "5", "4", None],
                   "6": ["5", "8", "9", "-", "+", None, "3", "2"],
                   "9": ["8", "=", "/", "*", "-", "+", "6", "5"],
                   "8": ["7", None, "=", "/", "9", "6", "5", "4"],
                   "=": [None, None, None, None, "/", "9", "8", "7"]},
    "keypad": {"+": ["9", "*", "-", None, None, None, None, "6"],
               "*": ["/", None, None, None, "-", "+", "9", "8"],
               "-": ["*", None, None, None, None, None, "+", "9"],
               "/": [None, None, None, None, "*", "9", "8", "7"],
               ".": ["0", "2", "3", None, None, None, None, None],
               "1": [None, None, "4", "5", "2", "0", None, None],
               "0": [None, "1", "2", "3", ".", None, None, None],
               "3": ["2", "5", "6", None, None, None, ".", "0"],
               "2": ["1", "4", "5", "6", "3", ".", "0", None],
               "5": ["4", "7", "8", "9", "6", "3", "2", "1"],
               "4": [None, None, "7", "8", "5", "2", "1", None],
               "7": [None, None, None, "/", "8", "5", "4", None],
               "6": ["5", "8", "9", "+", None, None, "3", "2"],
               "9": ["8", "/", "*", "-", "+", None, "6", "5"],
               "8": ["7", None, "/", "*", "9", "6", "5", "4"]}}
