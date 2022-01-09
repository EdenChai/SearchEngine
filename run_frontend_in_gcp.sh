##############################  CREATE INSTANCE  ##############################

# 0. Install Cloud SDK on your local machine or using Could Shell
# check that you have a proper active account listed
gcloud auth list 
# check that the right project and zone are active
gcloud config list
# if not set them
# gcloud config set project $PROJECT_NAME
# gcloud config set compute/zone $ZONE

# 1. Set up public IP
gcloud compute addresses create search-engine-337619-ip --project=search-engine-337619 --region=us-central1
gcloud compute addresses list

# note the IP address printed above, that's your extrenal IP address. Enter it here:
"INSTANCE_IP=35.202.162.205"

# 2. Create Firewall rule to allow traffic to port 8080 on the instance
gcloud compute firewall-rules create default-allow-http-8080 --allow tcp:8080 --source-ranges 0.0.0.0/0 --target-tags http-server

# 3. Create the instance. Change to a larger instance (larger than e2-micro) as needed.
gcloud compute instances create ir-project \
      --zone=us-central1-c --machine-type=e2-standard-4 \
      --network-interface=address=35.202.162.205, network-tier=PREMIUM, subnet=default \
      --metadata-from-file startup-script=startup_script_gcp.sh \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --tags=http-server --boot-disk-size 100GB

# monitor instance creation log using this command. When done (4-5 minutes) terminate using Ctrl+C
gcloud compute instances tail-serial-port-output ir-project --zone us-central1-c

# 4. Secure copy your app to the VM
gcloud compute scp /search_frontend.py edench@ir-project:/home/edench

# 5. SSH to your VM and start the app
gcloud compute ssh edench@ir-project
python3 search_frontend.py

##############################  DELETE INSTANCE  ##############################

# 1. Clean up commands to undo the above set up and avoid unnecessary charges
gcloud compute instances delete -q ir-project

# 2. make sure there are no lingering instances
gcloud compute instances list

# 3. delete firewall rule
gcloud compute firewall-rules delete -q default-allow-http-8080

# 4. delete external addresses
gcloud compute addresses delete -q search-engine-337619-ip --region us-central1