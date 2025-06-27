import sys
from src.logger import logging

def error_message_details(error, error_detail: sys):
    """
    Purpose: Extracts detailed information about an error for logging.
    
    Parameters:
    - error: The exception object.
    - error_detail: sys module to access traceback information.
    
    Returns:
    - str: A formatted string containing the error type, message, and traceback.
    """
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(file_name, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    """
    Custom exception class for handling errors in the application.
    Inherits from the built-in Exception class and overrides the constructor to log error details.
    """
    
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)
        # logging.error(self.error_message)
    
    def __str__(self):
        return self.error_message
