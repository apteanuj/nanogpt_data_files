place fineweb10B folder inside the data folder for nanogpt speedrun repo
merge chunks back using the one-liner : 
for base in $(ls *.part* | sed 's/\.part.*//' | sort -u); do cat "$base".part* > "$base" && rm "$base".part*; done