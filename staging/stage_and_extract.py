from stager_access import stage, get_status, get_webdav_urls_requested, get_macaroons
import sys
import os
import time

# Usage: python stage_and_extract.py <srm_list.txt> <output_directory>
if len(sys.argv) != 3:
    print("Usage: python stage_and_extract.py <srm_list.txt> <output_directory>")
    sys.exit(1)

srm_file = sys.argv[1]
output_dir = sys.argv[2]

# Check that the SRM file exists
if not os.path.isfile(srm_file):
    print(f"Error: SRM list file '{srm_file}' not found.")
    sys.exit(1)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read SRM URLs from file, stripping whitespace and skipping empty lines
with open(srm_file, 'r') as f:
    surls = [line.strip() for line in f if line.strip()]

# Submit staging request
print("Submitting staging request...")
stageid = stage(surls)
print(f"Stage ID: {stageid}")

# Poll for staging completion
tries = 0
max_tries = 1440 # Maximum wait one day for data to be staged
final_states = {"success", "failed", "aborted"}

while True:
    status = get_status(stageid)
    state = status if isinstance(status, str) else status.get("status", "unknown")
    print(f"[try {tries}] Status: {state}")

    if state.lower() in final_states:
        break

    tries += 1
    if tries >= max_tries:
        print("Max retries exceeded.")
        sys.exit(1)

    time.sleep(60)

# If successful, extract WebDAV URLs and macaroon token and save them
if state.lower() == "success":
    webdav_urls = get_webdav_urls_requested(stageid)
    macaroons = get_macaroons(stageid)
    token = next(iter(macaroons[0].values()))

    webdav_path = os.path.join(output_dir, "webdav_links.txt")
    macaroon_path = os.path.join(output_dir, "macaroon.txt")

    # Write WebDAV URLs to file
    with open(webdav_path, "w") as f:
        for url in webdav_urls:
            f.write(url + "\n")

    # Write macaroon token to file
    with open(macaroon_path, "w") as f:
        f.write(token)

    print("Staging complete.")
    print(f"Saved WebDAV links to {webdav_path}")
    print(f"Saved macaroon token to {macaroon_path}")

else:
    print(f"Staging failed: status = {state}")
    sys.exit(1)