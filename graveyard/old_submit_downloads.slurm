#!/bin/bash
#SBATCH --job-name=submit_lotaas_downloads  # Job name
#SBATCH --output=logs/submit_%j.log        # Log output
#SBATCH --error=logs/submit_%j.log         # Error log
#SBATCH --time=00:10:00                    # Small allocation for submission
#SBATCH --mem=1G                           # Minimal memory needed for submission

# Ensure logs directory exists
mkdir -p logs

# Input file with WebDAV URLs
INPUT_FILE="$1"
MACAROON="$2"  # Macaroon token as second argument

# Validate input
if [ -z "$INPUT_FILE" ] || [ -z "$MACAROON" ]; then
    echo "Usage: sbatch submit_downloads.sh <input_file> <macaroon_token>"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Create a temporary cleaned-up file
CLEANED_INPUT_FILE="${INPUT_FILE}_cleaned"

# Remove empty lines and carriage returns
sed '/^$/d' "$INPUT_FILE" | tr -d '\r' > "$CLEANED_INPUT_FILE"

# Read and submit jobs for each WebDAV URL
while IFS= read -r WEBDAV_URL || [[ -n "$WEBDAV_URL" ]]; do
    # Trim spaces
    WEBDAV_URL=$(echo "$WEBDAV_URL" | xargs)
    
    # Ensure URL is not empty
    if [[ -n "$WEBDAV_URL" ]]; then
        echo "Submitting job for: $WEBDAV_URL"
        sbatch download_lotaas.slurm "$WEBDAV_URL" $MACAROON
        echo "download_lotaas.slurm $WEBDAV_URL $MACAROON"
    fi
done < "$CLEANED_INPUT_FILE"

# Remove temporary cleaned-up file
rm "$CLEANED_INPUT_FILE"

echo "All jobs submitted."