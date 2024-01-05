from pytube import YouTube

# Replace the URL with the URL of the video you want to download
url = "https://www.youtube.com/watch?v=FsGPxhidwGg"

# Create a YouTube object
yt = YouTube(url)

# Get the highest resolution stream
stream = yt.streams.get_highest_resolution()

# Download the video
stream.download()
