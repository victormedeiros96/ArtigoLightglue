import requests
import json
import os
import time
from urllib.parse import quote

CVAT_URL = "http://192.168.18.140:8080"
USERNAME = "superrdt"
PASSWORD = "superpwdrdt"
PROJECT_ID = 79
FORMAT_NAME = "CVAT for images 1.1" # Standard tracking format
DOWNLOAD_DIR = "/mnt/hd2/ArtigoLightglue/Datasets_externos/cvat_plates"

def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    session = requests.Session()
    
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
    
    print(f"Fetching tasks for project {PROJECT_ID}...")
    tasks_response = session.get(f"{CVAT_URL}/api/tasks?project_id={PROJECT_ID}&page_size=100")
    tasks = tasks_response.json().get('results', [])
    
    done_tasks = [t for t in tasks if t.get('status') == 'completed']
    print(f"Found {len(done_tasks)} completed tasks.")
    
    for t in done_tasks:
        task_id = t['id']
        task_name = t['name']
        print(f"\nDownloading dataset for task {task_id}: {task_name}")
        
        # 1. Trigger export using POST
        export_url = f"{CVAT_URL}/api/tasks/{task_id}/dataset/export"
        payload = {"format": FORMAT_NAME, "save_images": True}
        resp = session.post(export_url, json=payload)
        
        if resp.status_code not in [201, 202]:
            print(f" -> Failed to start export: {resp.status_code} {resp.text}")
            continue
            
        rq_id = resp.json().get('rq_id')
        if not rq_id:
            print(f" -> No rq_id returned: {resp.text}")
            continue
            
        print(f" -> Export started (rq_id: {rq_id}). Waiting for completion...")
        
        # 2. Poll status
        result_url = None
        while True:
            poll_resp = session.get(f"{CVAT_URL}/api/requests/{rq_id}")
            if poll_resp.status_code != 200:
                print(f" -> Polling failed: {poll_resp.status_code} {poll_resp.text}")
                break
            
            status_data = poll_resp.json()
            state = status_data.get('state')
            if state == 'finished':
                result_url = status_data.get('result_url')
                break
            elif state == 'failed':
                print(f" -> Export failed on server: {status_data.get('message')}")
                break
                
            time.sleep(5)
            
        # 3. Download
        if result_url:
            download_resp = session.get(f"{CVAT_URL}{result_url}")
            if download_resp.status_code == 200:
                out_path = os.path.join(DOWNLOAD_DIR, f"{task_name}.zip")
                with open(out_path, 'wb') as f:
                    f.write(download_resp.content)
                print(f" -> Saved to {out_path}")
            else:
                print(f" -> Download failed: {download_resp.status_code} {download_resp.text}")

if __name__ == "__main__":
    main()
