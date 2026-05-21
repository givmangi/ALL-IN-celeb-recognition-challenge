import requests
import json
import datetime
import os

def submit_and_log(res_dict, model_name, group_name = "ALL-IN", url ="", log_file="/logs/submission_log.txt"):
    """
    make sure you load res_dict, which is the dictionary containing the results
    and model_name, which is the name of the model you are submitting (e.g., "RESNET50_split", "ArcFace_fulltrain", etc.)
    Note: url will be given by prof, make sure to change it asap so we don't forget :)
    Note 2: log_file is an example path I put, you can change it howevever you want or keep it as is xoxo
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Connecting to server to submit '{group_name}'...")
    
    payload = {
        "groupname": group_name,
        "results": res_dict
    }
    server_response_text = ""
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        server_response_text = response.text.strip()
        print(f"\nSUCCESS, Server responded: {server_response_text}\n")
    except requests.exceptions.HTTPError as http_err:
        server_response_text = f"HTTP Error: {http_err} - Server said: {response.text}"
        print(f"\nERROR: {server_response_text}\n")
    except Exception as err:
        server_response_text = f"Connection FAILED: {err}"
        print(f"\nERROR: {server_response_text}\n")
    log_entry = (
        f"Time:\t {timestamp}\n"
        f"Group:\t {group_name}\n"
        f"Model:\t {model_name}\n"
        f"Result:\t {server_response_text}\n"
        f"{'-'*60}\n"
    )
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        print(f"Log appended to: {log_file}")
    except Exception as e:
        print(f"Warning: Could not write to log file. Error: {e}")