import requests
import json
import os

CVAT_URL = "http://192.168.18.140:8080"
USERNAME = "superrdt"
PASSWORD = "superpwdrdt"
PROJECT_ID = 79

def main():
    session = requests.Session()
    
    # 1. Login
    print("Logging into CVAT...")
    auth_response = session.post(f"{CVAT_URL}/api/auth/login", json={
        "username": USERNAME,
        "password": PASSWORD
    })
    
    if auth_response.status_code != 200:
        print(f"Failed to login: {auth_response.text}")
        return
        
    token = auth_response.json().get('key')
    session.headers.update({"Authorization": f"Token {token}"})
    
    # 2. Get tasks for project 79
    print(f"Fetching tasks for project {PROJECT_ID}...")
    tasks_url = f"{CVAT_URL}/api/tasks?project_id={PROJECT_ID}&page_size=100"
    tasks_response = session.get(tasks_url)
    
    if tasks_response.status_code != 200:
        print(f"Failed to fetch tasks: {tasks_response.text}")
        return
        
    tasks = tasks_response.json().get('results', [])
    done_tasks = [t for t in tasks if t.get('status') == 'completed' or t.get('state') == 'completed'] # CVAT might use status=completed
    
    # Check 'status' field. CVAT v2 usually has 'status': 'completed' or 'annotation' etc.
    # Let's print the fields to be sure
    print(f"Total tasks found: {len(tasks)}")
    for t in tasks[:5]:
        print(f"Task {t['id']} - Name: {t['name']}, Status: {t.get('status')}")
        
    done_tasks = [t for t in tasks if t.get('status') == 'completed']
    print(f"\nTasks with status 'completed': {len(done_tasks)}")
    
    for t in done_tasks:
        print(f" - [{t['id']}] {t['name']}")

if __name__ == "__main__":
    main()
