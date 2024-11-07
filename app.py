import os
from googleapiclient.discovery import build
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv('API_KEY')  # API key from .env file
PORT = int(os.getenv('PORT', 5000)) # Access PORT from .env file or use default
app = Flask(__name__)
CORS(app)
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Lazy-loading Sentiment Analysis Pipeline to save memory
sentiment_pipeline = None

def get_sentiment_pipeline():
    global sentiment_pipeline
    if sentiment_pipeline is None:
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
    return sentiment_pipeline

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/youtube/<channel_id>', methods=['POST'])
def fetch_channel_data(channel_id):
    try:
        channel_response = youtube.channels().list(
            id=channel_id,
            part='snippet,statistics'
        ).execute()

        if 'items' not in channel_response or not channel_response['items']:
            return jsonify({'error': 'No channel found with the provided ID.'}), 404

        channel_stats = [{
            'Channel_name': channel_response['items'][0]['snippet']['title'],
            'Subscribers': int(channel_response['items'][0]['statistics'].get('subscriberCount', 0)),
            'Views': int(channel_response['items'][0]['statistics'].get('viewCount', 0)),
            'Total_videos': int(channel_response['items'][0]['statistics'].get('videoCount', 0)),
        }]

        video_ids = []
        next_page_token = None
        while True:
            video_response = youtube.search().list(
                channelId=channel_id,
                part='snippet',
                order='date',
                maxResults=50,
                pageToken=next_page_token
            ).execute()

            for item in video_response.get('items', []):
                if 'videoId' in item['id']:
                    video_ids.append(item['id']['videoId'])

            next_page_token = video_response.get('nextPageToken')
            if not next_page_token:
                break

        if not video_ids:
            return jsonify({'error': 'No videos found for this channel.'}), 404

        video_details = []
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i:i+50]
            videos = youtube.videos().list(
                id=','.join(chunk),
                part='snippet,statistics'
            ).execute()

            for video in videos['items']:
                snippet = video['snippet']
                stats = video['statistics']
                video_data = {
                    'Title': snippet['title'],
                    'Published_date': snippet['publishedAt'],
                    'Views': int(stats.get('viewCount', 0)),
                    'Likes': int(stats.get('likeCount', 0)),
                    'Comments': int(stats.get('commentCount', 0)),
                }
                video_details.append(video_data)

        return jsonify({'channel_stats': channel_stats, 'video_details': video_details})

    except Exception as e:
        print(f"Error fetching channel data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/youtube/search/<search_term>', methods=['POST'])
def search_top_videos(search_term):
    try:
        search_response = youtube.search().list(
            q=search_term,
            part='snippet',
            type='video',
            maxResults=10
        ).execute()

        top_videos = []
        for item in search_response['items']:
            video_data = {
                'Title': item['snippet']['title'],
                'Published_date': item['snippet']['publishedAt'],
                'Thumbnail': item['snippet']['thumbnails']['high']['url'],
                'videoId': item['id']['videoId'],
            }
            top_videos.append(video_data)

        return jsonify({'top_videos': top_videos})

    except Exception as e:
        print(f"Error fetching video data: {e}")
        return jsonify({'error': 'Error fetching video data'}), 500

@app.route('/youtube/comments/<video_id>', methods=['POST'])
def analyze_video_comments(video_id):
    try:
        comments = fetch_all_comments(video_id)
        total_comments = len(comments)

        if total_comments == 0:
            return jsonify({
                'total_comments': total_comments,
                'positive_comments': 0,
                'negative_comments': 0
            })

        analysis_results = analyze_comments(comments)

        # Initialize counts to 0 in case analysis_results is empty
        positive_comments = sum(1 for result in analysis_results if result['sentiment_category'] == 'Positive') if analysis_results else 0
        negative_comments = sum(1 for result in analysis_results if result['sentiment_category'] == 'Negative') if analysis_results else 0

        return jsonify({
            'total_comments': total_comments,
            'positive_comments': positive_comments,
            'negative_comments': negative_comments
        })

    except Exception as e:
        print(f"Error analyzing comments: {e}")
        return jsonify({'error': str(e)}), 500


def fetch_all_comments(video_id):
    comments = []
    try:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100
        ).execute()
        
        while response:
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            if 'nextPageToken' in response:
                response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat='plainText',
                    maxResults=100,
                    pageToken=response['nextPageToken']
                ).execute()
            else:
                break

    except Exception as e:
        print(f"Error fetching comments: {e}")
    
    return comments

def analyze_comments(comments):
    results = []
    max_length = 512

    for comment in comments:
        truncated_comment = comment[:max_length]
        
        try:
            sentiment = get_sentiment_pipeline()(truncated_comment)[0]
            sentiment_label = sentiment.get('label')
            sentiment_score = sentiment.get('score')

            if sentiment_label in ['0 stars', '1 star']:
                sentiment_category = 'Negative'
            elif sentiment_label == '2 stars':
                sentiment_category = 'Neutral'
            else:
                sentiment_category = 'Positive'

            results.append({
                'comment': truncated_comment,
                'sentiment': sentiment_label,
                'score': sentiment_score,
                'sentiment_category': sentiment_category
            })
        
        except Exception as e:
            print(f"Error analyzing comment: {e}")
            continue

    return results

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


