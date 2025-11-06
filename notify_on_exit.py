# -*- coding: utf-8 -*-
"""
A Python module to send an email notification when a script finishes execution.

This module uses the 'atexit' standard library to register a hook
that runs when the Python interpreter exits.

It calculates the total runtime and sends an email with execution statistics.
"""

import smtplib
import atexit
import time
import datetime
import sys
import os
from email.message import EmailMessage

# --- Private Functions ---

def _send_email(smtp_config, recipient_email, program_name, runtime_seconds, end_time_local):
    """
    Connects to the SMTP server and sends the notification email.
    
    Uses SMTP_SSL for a secure connection.
    """
    try:
        # 1. Create the Email Message
        msg = EmailMessage()
        msg['Subject'] = f"[Program Finished] {program_name}"
        msg['From'] = smtp_config['sender_email']
        msg['To'] = recipient_email
        
        # 2. Construct the email body
        # Using f-string for clear formatting.
        body = f"""
        Program execution has completed.

        --- Execution Stats ---
        Program Name:     {program_name}
        Total Runtime:    {runtime_seconds:.4f} seconds
        Completion Time:  {end_time_local} (Local Time)
        """
        
        msg.set_content(body.strip())

        # 3. Send the email
        # We use SMTP_SSL for ports like 465 (Gmail, etc.)
        # This creates a secure context automatically.
        print(f"\nConnecting to {smtp_config['server']}:{smtp_config['port']} to send notification...")
        
        with smtplib.SMTP_SSL(smtp_config['server'], smtp_config['port']) as server:
            # Login to the email server
            server.login(smtp_config['sender_email'], smtp_config['sender_password'])
            
            # Send the message
            server.send_message(msg)
            
        # Use sys.stderr to avoid mixing with main program's stdout
        print(f"Notification email successfully sent to {recipient_email}", file=sys.stderr)

    except Exception as e:
        # If email fails, we are exiting anyway.
        # Print the error to standard error.
        print(f"CRITICAL: Failed to send notification email. Error: {e}", file=sys.stderr)

# --- Public API ---

def register_completion_notify(recipient_email, smtp_config, program_name=None):
    """
    Registers the notification function to be called at script exit.

    This is the main function you should call from your script.
    Call this function *early* in your main script.

    Args:
        recipient_email (str): The email address to send the notification to.
        
        smtp_config (dict): A dictionary with SMTP server details.
            Expected keys:
            - 'server' (str): e.g., "smtp.gmail.com"
            - 'port' (int): e.g., 465 (for SSL)
            - 'sender_email' (str): Your full email address.
            - 'sender_password' (str): Your email password or App Password.
                                      (App Password is required for Gmail/Google).
                                      
        program_name (str, optional): The name of the program. 
                                    Defaults to the basename of the script (sys.argv[0]).
    """
    
    # 1. Record the start time immediately when registered
    start_time = time.time()
    
    # 2. Determine the program name
    # If not provided, use the filename of the main script
    if program_name is None:
        program_name = os.path.basename(sys.argv[0])

    # 3. Define the internal function that will run at exit
    def _notify_at_exit():
        """
        This function is executed by the 'atexit' module.
        It calculates runtime statistics and triggers the email.
        """
        
        # Calculate final statistics
        end_time = time.time()
        runtime_seconds = end_time - start_time
        end_time_local = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')
        
        print(f"--- Program '{program_name}' finishing. Runtime: {runtime_seconds:.4f}s ---")
        
        # Call the email sending function with all collected data
        _send_email(
            smtp_config, 
            recipient_email, 
            program_name, 
            runtime_seconds, 
            end_time_local
        )

    # 4. Register the function with atexit
    atexit.register(_notify_at_exit)
    
    print(f"Email notification registered. Will notify {recipient_email} on exit.", file=sys.stderr)
