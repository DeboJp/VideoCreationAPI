#requestAPITest.py
import requests 

def download_music(url, filename="music.mp3"):
    """
    Downloads a music file from a given URL and saves it with the specified filename. 
    
    Args:
        url (str): The URL of the music file.
        filename (str, optional): The desired filename for the downloaded file. Defaults to "music.mp3". 
    """
    response = requests.get(url, stream=True) 
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk) 
        print(f"Music file downloaded as: {filename}")
    else:
        print("Error downloading music file") 
    
# Example usage 
music_url = "https://cdn1.suno.ai/9ef53071-ef0e-4d23-b402-fc2f04dbd3bb.mp3" 
download_music(music_url) 

def download_image(url, filename="image.jpg"):
    """
    Downloads an image file from a given URL and saves it with the specified filename.

    Args:
        url (str): The URL of the image file.
        filename (str, optional): The desired filename for the downloaded image. Defaults to "image.jpg".
    """
    import requests  # Import requests module for making HTTP requests.

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Image file downloaded as: {filename}")
    else:
        print("Error downloading image file")

# Example usage
image_url = "https://cdn1.suno.ai/image_large_9ef53071-ef0e-4d23-b402-fc2f04dbd3bb.jpeg"
download_image(image_url)
