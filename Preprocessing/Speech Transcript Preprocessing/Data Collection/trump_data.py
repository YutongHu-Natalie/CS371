import requests
from bs4 import BeautifulSoup
import re
import csv

def get_transcript_links(base_url, num_pages):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.77 Safari/537.36'
    }
    links = []

    for page_num in range(1, num_pages + 1):
        url = f'{base_url}/page/{page_num}'
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            divs = soup.find_all('div', class_='fl-post-grid-post')

            for div in divs:
                a_tag = div.find('a', href=True)
                if a_tag:
                    links.append(a_tag['href'])
                    print(f"Page {page_num} - Link: {a_tag['href']}")
        else:
            print(f"Failed to retrieve page {page_num}. Status: {response.status_code}")

    return links

def extract_transcript(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.77 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'No Title Found'

        # Extract date using regex
        date_pattern = re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4}')
        all_text = soup.get_text()
        date = date_pattern.search(all_text).group() if date_pattern.search(all_text) else 'No Date Found'

        # Extract transcript
        paragraphs = soup.find_all('p')
        transcript = []
        start_extracting = False
        transcribe_marker_count = 0

        for p in paragraphs:
            text = p.get_text()

            if "2023" in text or "2024" in text or "2022" in text or "2021" in text or "2020" in text or "2019" in text:
                start_extracting = True

            if "Transcribe Your Own Content" in text:
                transcribe_marker_count += 1
                if transcribe_marker_count == 1:
                    print("Encountered first 'Transcribe Your Own Content'. Ignoring.")
                    continue  # Ignore the first occurrence
                elif transcribe_marker_count == 2:
                    print("Encountered second 'Transcribe Your Own Content'. Stopping extraction.")
                    break  # Stop at the second occurrence

            if start_extracting:
                transcript.append(text)

        return title, date, "\n".join(transcript), url
    else:
        print(f"Failed to retrieve transcript from {url}. Status: {response.status_code}")
        return None, None, None, None

def save_to_csv(filename, data):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Date', 'Transcript', 'URL'])
        for row in data:
            writer.writerow([element.strip() if isinstance(element, str) else element for element in row])

# Main script
base_url = 'https://www.rev.com/blog/transcript-category/donald-trump-transcripts'
links = get_transcript_links(base_url, num_pages=40)

data = []
for link in links:
    print(f"Extracting transcript from: {link}")
    title, date, transcript, url = extract_transcript(link)
    if transcript:
        print(f"Title: {title}\nDate: {date}")
        data.append([title, date, transcript, url])

# Save to CSV
if data:
    save_to_csv('donald_trump_transcripts.csv', data)
    print("Transcripts saved to transcripts.csv")
else:
    print("No transcript data available.")
