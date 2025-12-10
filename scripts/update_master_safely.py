import json
import os

file_path = "MASTER_DATA.json"

new_items = [
    {
        "id": "campus_wifi_procedure",
        "content": "## NIT WARANGAL WI-FI CONNECTIVITY PROCEDURE\nWe have two Wi-Fi SSIDs in Campus network:\n1. Provisioning\n2. NITW\n\n**Android Devices Connection Process:**\n\n**Step 1:**\nInstall 'Clearpass Quick Connect' app from Google Play store. Grant necessary permissions after installation.\n\n**Step 2:**\nConnect to 'Provisioning' SSID. Click 'Sign in to WLan Network'. Use Campus mail-id. A password will pop up on your mobile.\n*Note:* If 'Sign in to WLAN network' pop-up doesn't appear, open Google Chrome and visit: `https://aaa-1.nitw.ac.in/guest/Provisioning_Onboarding.php`\n(Note: 'Provisioning' has a Capital P, 'Onboarding' has a Capital O).\n\n**Step 3:**\nSelect 'I have already installed Quick Connect'. Download the network profile.\nOpen the downloaded file using 'QuickConnect'.\nAfter clicking `quick1x.networkconfig`, it will successfully connect to NITW network and start provisioning.\n\n**Final Step:**\nAfter successfully completing the above process, connect to 'NITW' SSID only (not Provisioning).\nThis is a one-time process. Next time onwards, connect directly to 'NITW'.",
        "metadata": {
            "category": "Facilities",
            "sub_category": "Wi-Fi",
            "filter": ""
        }
    },
    {
        "id": "sports_and_gym_facilities",
        "content": "## FACILITIES AND PROGRAMMES OF THE DEPARTMENT (Sports & Gym)\n\n## Facilities Overview\nThe department has Indoor (fitness center/gym, wooden flooring for shuttle badminton) and outdoor sports facilities in almost all disciplines (400mtrs track, flood lit courts, stadium pavilion) except swimming pool.\n\n## Programs & Competitions\nThe department will organize intramural competitions branch wise in all the disciplines to give an opportunity to involve more number of student community.\nThe department will provide an opportunity to talented sportsmen and women to participate in the extramural competitions like, Inter NIT sports meet and Inter University sports competitions regularly.\nThe department will organize sports tournaments at our campus regularly.\nFitness and Conditioning classes were conducted to all first year students regularly during academic year.\n\n## List of Sports Events\nThe sports comprises of the following events:\n* Athletics\n* Cricket\n* Kabaddi\n* Basketball\n* Volleyball\n* Badminton\n* Weight Lifting\n* Best Physique\n* Tennis\n* Football\n* Ball badminton\n* Chess\n* Table - Tennis\n* Kho-Kho\n* Tenni – Koit\n* Throwball\n\n## Gym and Sports Timings\nTimings: 5:00 am to 8.30 a.m. and 4:00 p.m. to 9.00 p.m.\nOpen to the students apart from office time.",
        "metadata": {
            "category": "Facilities",
            "sub_category": "Sports",
            "filter": ""
        }
    }
]

def update_data():
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} items.")
        
        # Remove existing items with same ID if any (to avoid duplicates during retries)
        existing_ids = {item["id"] for item in new_items}
        data = [item for item in data if item.get("id") not in existing_ids]
        
        # Append new items
        data.extend(new_items)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
        print(f"Successfully updated {file_path}. Total items: {len(data)}")
        
    except Exception as e:
        print(f"Error updating file: {e}")

if __name__ == "__main__":
    update_data()
