def consoleLog(message: str, method: str, isError: bool = False):
    colorEscape = "\033[91m" if isError else "\033[1;32m"
    
    print(colorEscape + "[CONSOLE - " + method.upper() + "]\033[0m", message)